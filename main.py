import argparse
from collections import defaultdict
from typing import Union, Tuple

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train.preliminary import train_model_preliminary
from utils.data_preprocess import read_dataset, select_dataset, preprocess_dataset, split_dataset, \
    PreliminaryDataset, read_unlabeled_dataset
import utils.tokenization
from module.MT5 import MT5ForConditionalGeneration
# from transformers import MT5ForConditionalGeneration, AutoTokenizer
# from transformers import AutoModel

from utils.tokenization import tokenizer_dataset

from utils.ExtraNameSpace import NameSpace

import higher


def args_parse():
    parser = argparse.ArgumentParser(description="semantic_parser")

    parser.add_argument('--config', type=str, default='config.yaml',
                    help='config file, 只使用单层的嵌套')

    parser.add_argument("--dataset", type=str, default="ours",
                        choices=["ours", "topv2"])

    parser.add_argument("--train_dataset_dir", type=str, default="./data/dev",
                    help="train dataset dir")

    parser.add_argument("--unlabel_dataset_dir", type=str, default="./data/dev",
                    help="unlabel dataset dir")

    parser.add_argument("--task", type=str, default="preliminary", choices=["preliminary", "self-train"],
                    help="省得分文件了")

    parser.add_argument("--model_dir", type=str, default="/hy-tmp/mt5_small",
                    help="model dir")

    parser.add_argument("--save_dir", type=str, default="/hy-tmp/t5",
                    help="save dir")

    parser.add_argument("--experiment_name", type=str, default="Default", choices=["Default"], # 这个比如10样例、100样例等
                    help="experiment name")

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

    parser.add_argument("--batch_size", type=int, default=1,
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

    if args.config:
        yaml_config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        # 合并yaml到args里面
        for key, value in yaml_config.items():
            setattr(args, key, value)
    #         parser.add_argument(f'--{key}', default=value, help=f'{key} from YAML config')

    NameSpace._args = args

    return args

def get_dataset(tokenizer, args) -> dict:
    """
    :return: 要么返回dict{op: Dataset}, 要么返回{'train': Dataloader}这种
    我有点纠结的是，一般来说返回Dataloader足够了，但预实验最多只能到tokenized_dataset这一步，是不是有必要把self-train也回退到这步呢？
    虽然看着整齐点但好像意义不大，先就这样错开吧
    """
    dataset = read_dataset(args.train_dataset_dir)

    if args.task == "preliminary":# 预实验时候默认没有测试集
        selected_dataset: dict = select_dataset(dataset, args) # 只有预实验需要区分训练集和测试集，并且每个operator单独
        # 维护一个dataloader
        final_dataset = {}
        for op in selected_dataset:
            # 先不划分训练测试，因为要多轮迭代着拆分
            final_dataset[op] = tokenizer_dataset(tokenizer,
                                                  preprocess_dataset(selected_dataset[op]))

        return final_dataset

    elif args.task == "self-train":
        # 非预实验的时候，有个拆分或者按某个实验标准进行分割的过程，不过这个放在别处也行
        # self train需要正常训练测试+一个无标签数据集，所以需要额外一个读入或者无标签的读入是从训练集or验证集里拆出来的
        dataset["unlabeled"] = read_unlabeled_dataset(args.unlabel_dataset_dir)
        # 然后self train还有个保存和读入topk的环节，但这个应该也是边训边存

        for key in dataset:
            dataset[key] = tokenizer_dataset(tokenizer, preprocess_dataset(dataset[key]))

        dataset["unlabeled"] = dataset["unlabeled"]["input_ids"]

        return dataset

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



def main():
    args = args_parse()

    #=====test

    model = MT5ForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    summary = "Weiter Verhandlung in Syrien."
    inputs = tokenizer(article, text_target=summary, return_tensors="pt")

    outputs = model(**inputs)
    loss = outputs.loss
    #=====test

    model = MT5ForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.to(args.device)
    model.to(args.device)

    dataset = get_dataset(tokenizer, args)

    optimizer = get_optimizer(args.optimizer, model, args)
    criterion = get_criterion(args)


    if args.task == "preliminary":
        for op in dataset:
            print("算子是", op)
            train_model_preliminary(model, optimizer, dataset[op], args)

    elif args.task == "self-train":
        train_model_self_train(model, optimizer, dataset, args)

if __name__ == '__main__':
    main()