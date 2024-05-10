import argparse
from collections import defaultdict
from typing import Union, Tuple

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, AutoTokenizer

from utils.data_preprocess import read_dataset_construct, select_dataset, preprocess_dataset, split_dataset, \
    PreliminaryDataset
from utils.tokenization_and_dataset import tokenizer_dataset, mycollate

import higher


def args_parse():
    parser = argparse.ArgumentParser(description="semantic_parser")

    parser.add_argument("--train_dataset_dir", type=str, default="./data/dev",
                    help="train dataset dir")

    parser.add_argument("--test_dataset_dir", type=str, default="./data/dev",
                    help="test dataset dir")

    parser.add_argument("--task", type=str, default="preliminary", choices=["preliminary", "self-train"],
                    help="省得分文件了")

    parser.add_argument("--model_dir", type=str, default="/home/lzx/T5-base-lora/model/mt5-base-trained-9",
                    help="model dir")

    parser.add_argument("--save_dir", type=str, default="/home/lzx/T5-base-lora/model/mt5-base-trained-10",
                    help="save dir")

    parser.add_argument("--optimizer", type=str, default="Adam",
                    help="optimizer")

    parser.add_argument("--lr", type=float, default=3e-5,
                    help="learning rate")

    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss",
                    help="criterion")

    parser.add_argument("--device", type=str, default="cuda",
                    help="device")

    parser.add_argument("--epoch", type=int, default=3,
                    help="epoch")

    parser.add_argument("--batch_size", type=int, default=16,
                    help="batch size")

    parser.add_argument("--operator_num", type=int, default=5,
                    help="做预实验的算子数量")

    parser.add_argument("--example_num", type=int, default=50,
                    help="每个算子做预实验的样本数量，包含训练和测试用的总数")

    parser.add_argument("--split", type=float, default=0.35,
                    help="训练集和测试集的划分比例")

    parser.add_argument("--iterations_per_subset", type=int, default=20,
                    help="对每个子集迭代20次")

    parser.add_argument("--select_top_percentage", type=float, default=0.8,
                    help="挑选子集的比例")

    parser.add_argument("--seed", type=int, default=192,
                    help="random seed")

    args = parser.parse_args()

    return args

def get_dataset(tokenizer, args) -> dict:
    """
    :return: 要么返回dict{op: Dataset}, 要么返回{'train': Dataloader}这种
    我有点纠结的是，一般来说返回Dataloader足够了，但预实验最多只能到tokenized_dataset这一步，是不是有必要把self-train也回退到这步呢？
    虽然看着整齐点但好像意义不大，先就这样错开吧
    """

    if args.task == "preliminary":
        dataset = read_dataset_construct(args.train_dataset_dir) # 预实验时候默认没有测试集
        selected_dataset: dict = select_dataset(dataset, args) # 只有预实验需要区分训练集和测试集，并且每个operator单独
        # 维护一个dataloader
        final_dataset = {}
        for op in selected_dataset:
            # 先不划分训练测试，因为要多轮迭代着拆分
            final_dataset[op] = tokenizer_dataset(tokenizer,
                                                  preprocess_dataset(selected_dataset[op]))

        return final_dataset

    elif args.task == "self-train":
        raise NotImplementedError("Not implemented yet.")

    raise ValueError(f"Unknown task: {args.task}")

def get_optimizer(optimizer, model, args):
    arguments = ['lr']
    input_args = {arg: getattr(args, arg) for arg in arguments}

    if optimizer == 'Adam':
        return optim.Adam(model.parameters(), **input_args)

    raise ValueError(f"Unknown optimizer: {optimizer}")

def get_criterion(args):
    if args.criterion == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()

    raise ValueError(f"Unknown criterion: {args.criterion}")

def train_model_preliminary(model, optimizer, tokenized_dataset, args):
    """
    :param model:
    :param optimizer:
    :param criterion:
    :param dataloader: 这个dataloader比较特殊，只是dict{op: }
    :param args:
    :return: 返回测试loss，或者考虑考虑别的
    """
    # 预留0.2的样本不参与元学习部分。这个参数比较随意，我就不放在args里了
    test_split_num = int(len(tokenized_dataset) * 0.8)
    sub_dataset, sub_test_dataset = tokenized_dataset[:test_split_num], tokenized_dataset[test_split_num:]
    dev_loss_for_example = defaultdict(list)  # 记录每个样例的测试loss
    example_to_data = {data["expression"]: data for data in sub_dataset} # 因为dev_loss_for_example是按照样例来记录的

    sub_dataset.shuffle()
    def get_sub_dataloader():
        train_dataset, _, dev_dataset = split_dataset(sub_dataset, args.split) # test相当于此刻的dev

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=mycollate)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=mycollate)

        return train_dataloader, dev_dataloader

    # 训练的时候不希望更改模型的初始参数
    def inner_train(dataloader1, dataloader2) -> float:
        whole_loss = 0

        with higher.innerloop_ctx(
                model,
                optimizer,
                copy_initial_weights=True,
                track_higher_grads=False,
        ) as (fmodel, diffopt):
            for epoch in range(args.epoch):
                for batch in dataloader1:
                    batch = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    diffopt.step(fmodel(**batch)["loss"])

            with torch.no_grad():
                for i, batch in enumerate(dataloader2):
                    batch = {k: v.to(args.device) for k, v in batch.items()}
                    outputs = fmodel(**batch)
                    whole_loss += outputs["loss"].detach().cpu()

        return whole_loss

    # 这里开始执行训练
    for _ in range(args.iterations_per_subset):
        train_dataloader, dev_dataloader = get_sub_dataloader()
        whole_loss = inner_train(train_dataloader, dev_dataloader)
        for i, batch in enumerate(dev_dataloader):
            # 遍历每个样例，记录测试loss
            for example in batch.expression:
                dev_loss_for_example[example].append(whole_loss)

    # 比较用全sub训练和sub的前80%训练的效果，在test上测试
    #1. 使用sub_dataset训练，sub_test_dataset测试
    sub_dataloader = DataLoader(sub_dataset, batch_size=args.batch_size, collate_fn=mycollate)
    sub_test_dataloader = DataLoader(sub_test_dataset, batch_size=args.batch_size, collate_fn=mycollate)

    loss_all_dataset = inner_train(sub_dataloader, sub_test_dataloader)

    #2. sub前80%
    for example in dev_loss_for_example:
        dev_loss_for_example[example] = sum(dev_loss_for_example[example]) / len(dev_loss_for_example[example])

    # 排序选loss最小的80%
    sorted_dev_loss = sorted(dev_loss_for_example.items(), key=lambda x: x[1])
    select_top_percentage = int(len(sorted_dev_loss) * args.select_top_percentage)
    selected_examples = [example_to_data[example] for example, _ in sorted_dev_loss[:select_top_percentage]]
    selected_dataset = PreliminaryDataset(*zip(*selected_examples))

    selected_dataloader = DataLoader(selected_dataset, batch_size=args.batch_size, collate_fn=mycollate)

    loss_select = inner_train(selected_dataloader, sub_test_dataloader)

    print("全数据微调的效果是：", loss_all_dataset)
    print("部分数据微调的效果：", loss_select)
    print("选择比例", args.select_top_percentage)
    print("=====================")

def main():
    args = args_parse()

    model = MT5ForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.to(args.device)

    dataset = get_dataset(tokenizer, args)

    optimizer = get_optimizer(args.optimizer, model, args)
    criterion = get_criterion(args)

    if args.task == "preliminary":
        for op in dataset:
            print("算子是", op)
            train_model_preliminary(model, optimizer, dataset[op], args)

if __name__ == '__main__':
    main()