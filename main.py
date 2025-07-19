import argparse
from datetime import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from train.ray_train import tune_hyperparameters_ray


from datasets import load_dataset

from test_model import test_model


import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

from train.preliminary import train_model_preliminary
from train.self_train import train_model_self_train
from utils.data_preprocess import read_dataset, select_dataset, preprocess_dataset, split_dataset, \
    PreliminaryDataset, read_unlabeled_dataset
import utils.tokenization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import AutoModel

from utils.tokenization import tokenizer_dataset

from utils.ExtraNameSpace import NameSpace

# import higher


def args_parse():
    parser = argparse.ArgumentParser(description="semantic_parser")

    parser.add_argument('--config', type=str, default='config.yaml',
                    help='config file, 只使用单层的嵌套')

    parser.add_argument("--dataset", type=str, default="topv2",
                        choices=["ours", "topv2", "zcl", "zcl_mixed"])

    parser.add_argument("--train_dataset_dir", type=str, default="./data/dev",
                    help="train dataset dir")

    parser.add_argument("--unlabel_dataset_path", type=str, default="./data/dev",
                    help="unlabel dataset dir")

    parser.add_argument("--test_dataset_dir", type=str, default="./data/dev",
                        help="test dataset dir")

    parser.add_argument("--task", type=str, default="self-train", choices=["preliminary", "self-train"],
                    help="省得分文件了")

    parser.add_argument("--close_selftrain", type=bool, default=False,
                        help="如果当前选项为True且task为self-train，则只训练基础模型，不进行自训练")

    parser.add_argument("--model_dir", type=str,
                        default="/data/pretrained_models/t5-base",#"/data/pretrained_models/t5-base",
    #,"/home/lzx/T5-base/model3/mt5-base-trained-final-500+500-2-7_again"
                    help="model dir")

    parser.add_argument("--save_dir", type=str, default="/home/lbq/data/semantic_parsing_711/output",
                    help="save dir")

    parser.add_argument("--experiment_name", type=str, default="Default", choices=["Default"], # 这个比如10样例、100样例等
                    help="experiment name")

    parser.add_argument("--optimizer", type=str, default="Adam",
                    help="optimizer")

    parser.add_argument("--lr", type=float, default=1e-5,
                    help="learning rate")

    parser.add_argument("--sf_lr", type=float, default=1e-5,
                    help="learning rate for self-train")

    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss",
                    help="criterion")

    parser.add_argument("--device", type=str, default="cuda",
                    help="device")

    parser.add_argument("--epoch", type=int, default=3,
                    help="epoch")

    parser.add_argument("--batch_size", type=int, default=128,
                    help="batch size")

    parser.add_argument("--max_length", type=int, default=128,
                    help="max length")

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

    parser.add_argument("--selftrain_iteration", type=int, default=10,
                        help="self train的迭代次数")

    parser.add_argument("--selftrain_topk", type=int, default=5,
                        help="self train的topk")

    parser.add_argument("--given_model", type=bool, default=False,
                        help="是否给定模型，如果是的话就直接训self train")

    args = parser.parse_args()

    # if args.config:
    #     yaml_config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    #     # 合并yaml到args里面
    #     for key, value in yaml_config.items():
    #         setattr(args, key, value)
    # #         parser.add_argument(f'--{key}', default=value, help=f'{key} from YAML config')
    #
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

        for key in dataset:
            print('key', key)
            dataset[key] = tokenizer_dataset(tokenizer, preprocess_dataset(dataset[key]))

        # self train需要正常训练测试+一个无标签数据集，所以需要额外一个读入或者无标签的读入是从训练集or验证集里拆出来的
        if not args.close_selftrain:
            dataset["unlabeled"] = read_unlabeled_dataset(args.unlabel_dataset_path)
        # 然后self train还有个保存和读入topk的环节，但这个应该也是边训边存

        # dataset["unlabeled"] = dataset["unlabeled"]["input_ids"]

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


def get_dataset_path():
    tasks = ['event', 'reminder', 'weather']
    dataset_types = ['top', 'our_data', 'our_add_top']
    exp_settings = ['SPIS25', 'SPIS50', 'full']

    gpus = 1  # torch count_devices是因为预先生成安全一些
    batch_small = round(8 / gpus)
    batch_middle = round(64 / gpus)
    batch_large = round(128 / gpus)

    data_dir = r'./original_data'
    data_dir = os.path.abspath(data_dir)

    for task in tasks:
        for dataset_type in dataset_types:
            for exp_setting in exp_settings:
                if dataset_type == 'our_data':
                    data_path = os.path.join(data_dir, task, dataset_type)
                else:
                    data_path = os.path.join(data_dir, task, dataset_type, exp_setting)

                if dataset_type == 'top':
                    batchsize = batch_small
                elif exp_setting == 'full':
                    batchsize = batch_large
                else:
                    batchsize = batch_middle

                unlabeled_path = os.path.join(data_dir, f"{task}_unlabel_train.tsv")

                yield data_path, batchsize, unlabeled_path


def main():
    args = args_parse()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    for path, batchsize, unlabeled_path in get_dataset_path():
        args.train_dataset_dir = path
        args.unlabel_dataset_path = unlabeled_path
        args.test_dataset_dir = path

        args.batch_size = batchsize

        args.close_selftrain = True
        # fixme: args.output (或者就不改了，每次测一下直接覆盖)
        # fixme: 开不开self train的都要来一遍
        # batchsize

        dataset = get_dataset(tokenizer, args)

        optimizer = get_optimizer(args.optimizer, model, args)
        criterion = get_criterion(args)

        if args.task == "preliminary":
            for op in dataset:
                print("算子是", op)
                train_model_preliminary(model, optimizer, dataset[op], args)

        elif args.task == "self-train":
            # train_model_self_train(model, tokenizer, optimizer, dataset, args)
            tune_hyperparameters_ray(tokenizer, dataset, args, args.save_dir, optimizer, model)

        dataset = preprocess_dataset(load_dataset(path))
        acc, f1 = test_model(model, tokenizer, dataset, device='cuda')
        # 保存指标到文件
        _save_metrics_to_file(acc, f1, args, path)
        print((path, acc, f1))


def _save_metrics_to_file(acc, f1, args, dataset_path):
    """
    将acc和f1指标存入文件
    :param acc: 准确率
    :param f1: F1分数
    :param args: 命令行参数
    :param dataset_path: 当前数据集路径（用于区分不同实验）
    """
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 生成带时间戳的文件名（避免重复）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{args.experiment_name}_{timestamp}.txt"
    file_path = os.path.join(args.save_dir, filename)

    # 写入指标内容（包含关键上下文信息）
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"===== 实验指标记录 =====\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验名称: {args.experiment_name}\n")
        f.write(f"数据集路径: {dataset_path}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"任务类型: {args.task}\n")
        f.write(f"准确率(Accuracy): {acc:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n")
        f.write(f"========================\n\n")

if __name__ == '__main__':
    main()
