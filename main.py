import argparse
from collections import defaultdict
from typing import Union, Tuple

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

import json
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

from train.preliminary import train_model_preliminary
from train.self_train import train_model_self_train
from utils.data_preprocess import read_dataset, select_dataset, preprocess_dataset, split_dataset, \
    PreliminaryDataset, read_unlabeled_dataset
import utils.tokenization
from module.MT5 import MT5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import AutoModel

from utils.tokenization import tokenizer_dataset

from utils.ExtraNameSpace import NameSpace

# import higher
import ray
from ray import tune
from ray.tune import CLIReporter


def args_parse():
    parser = argparse.ArgumentParser(description="semantic_parser")

    parser.add_argument('--config', type=str, default='config.yaml',
                    help='config file, 只使用单层的嵌套')

    parser.add_argument("--dataset", type=str, default="topv2",
                        choices=["ours", "topv2", "zcl", "zcl_mixed"])

    parser.add_argument("--train_dataset_dir", type=str, default="/home/lzx2000/temp/lzx/lzx/test/test/semantic_parsing_few_shot_our_2/TOPv2/low_resource_splits/our/data",
                    help="train dataset dir")

    parser.add_argument("--unlabel_dataset_dir", type=str, default="/home/lzx2000/temp/lzx/lzx/test/test/semantic_parsing_few_shot_2/TOPv2/low_resource_splits/unlabel",
                    help="unlabel dataset dir")

    parser.add_argument("--test_dataset_dir", type=str, default="./data/dev",
                        help="test dataset dir")

    parser.add_argument("--task", type=str, default="self-train", choices=["preliminary", "self-train"],
                    help="省得分文件了")

    parser.add_argument("--model_dir", type=str,
                        default="/data/pretrained_models/mt5-base/",#"/data/lbq/models/mt5-base-trained-final-500+500-2-7_again",
    #,"/home/lzx/T5-base/model3/mt5-base-trained-final-500+500-2-7_again"
                    help="model dir")

    parser.add_argument("--save_dir", type=str, default="./mt5-base",
                    help="save dir")

    parser.add_argument("--experiment_name", type=str, default="Default", choices=["Default"], # 这个比如10样例、100样例等
                    help="experiment name")

    parser.add_argument("--optimizer", type=str, default="AdamW",
                    help="optimizer")

    parser.add_argument("--lr", type=float, default=5e-5,
                    help="learning rate")

    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss",
                    help="criterion")

    parser.add_argument("--device", type=str, default="cuda",
                    help="device")

    parser.add_argument("--epoch", type=int, default=30,
                    help="epoch")

    parser.add_argument("--batch_size", type=int, default=2,
                    help="batch size")

    parser.add_argument("--max_length", type=int, default=128,
                    help="max length")

    parser.add_argument("--operator_num", type=int, default=5,
                    help="做预实验的算子数量")

    parser.add_argument("--example_num", type=int, default=50,
                    help="每个算子做预实验的样本数量，包含训练和测试用的总数")

    parser.add_argument("--split", type=float, default=0.05,
                    help="训练集和测试集的划分比例")

    parser.add_argument("--iterations_per_subset", type=int, default=20,
                    help="对每个子集迭代20次")

    parser.add_argument("--select_top_percentage", type=float, default=0.8,
                    help="挑选子集的比例")

    parser.add_argument("--seed", type=int, default=192,
                    help="random seed")

    parser.add_argument("--selftrain_iteration", type=int, default=2,
                        help="self train的迭代次数")

    parser.add_argument("--selftrain_topk", type=int, default=1,
                        help="self train的topk")

    parser.add_argument("--given_model", type=bool, default=False,
                        help="是否给定模型，如果是的话就直接训self train")

    parser.add_argument('--local-rank', type=int, default=-1, 
                        help="local rank for distributed training")

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
            dataset[key] = tokenizer_dataset(tokenizer, preprocess_dataset(dataset[key]))

        # self train需要正常训练测试+一个无标签数据集，所以需要额外一个读入或者无标签的读入是从训练集or验证集里拆出来的
        # dataset["unlabeled"] = read_unlabeled_dataset(args.unlabel_dataset_dir)
        # 然后self train还有个保存和读入topk的环节，但这个应该也是边训边存

        # dataset["unlabeled"] = dataset["unlabeled"]["input_ids"]

        return dataset

    raise ValueError(f"Unknown task: {args.task}")

def get_optimizer(optimizer, model, args):
    arguments = ['lr']
    input_args = {arg: getattr(args, arg) for arg in arguments}

    if optimizer == 'Adam':
        return optim.Adam(model.parameters(), **input_args)
    elif optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=args.lr)

    raise ValueError(f"Unknown optimizer: {optimizer}")

def get_criterion(args):
    if args.criterion == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()

    raise ValueError(f"Unknown criterion: {args.criterion}")


from copy import deepcopy
from testModel import test_model

def train_tune(config, args, tokenizer, dataset):
    """
    Ray Tune 的试验函数：根据 config 中的超参数训练模型、评估，并报告指标给 Tune。
    """
    # 更新超参数
    args.batch_size = config["batch_size"]
    args.learn_rate = config["learn_rate"]
    args.max_length = config["max_length"]
    args.epoch = config["epoch"]

    # 加载模型并设置设备
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.to(args.device)

    # 构造优化器（注意每个试验内部新建 optimizer）
    optimizer = get_optimizer(args.optimizer, model, args)
    
    # 训练模型
    model = train_model_self_train(model, tokenizer, optimizer, dataset, args)
    
    # 在训练后评估模型
    accuracy, avg_loss = test_model(model, tokenizer, dataset, args)
    
    # 报告指标给 Ray Tune（Ray Tune 会根据这些指标进行调度和选择最佳试验）
    tune.report({"accuracy": accuracy, "avg_loss": avg_loss})

def tune_hyperparameters_ray(tokenizer, dataset, args, model_save_path):
    """
    利用 Ray Tune 进行超参数网格搜索，每个试验分配一个 GPU。
    """
    # 定义超参数搜索空间
    config = {
        "batch_size": tune.grid_search([4, 8, 16, 32]),
        "learn_rate": tune.grid_search([1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5]),
        "max_length": tune.grid_search([32, 64, 128]),
        "epoch": tune.grid_search([8, 16, 32, 64])
    }
    
    # 设置一个 CLI 报告器，可以在命令行中看到进度
    reporter = CLIReporter(
        metric_columns=["accuracy", "avg_loss", "training_iteration"]
    )
    
    # 调用 tune.run 开始超参数搜索
    analysis = tune.run(
        tune.with_parameters(train_tune, args=args, tokenizer=tokenizer, dataset=dataset),
        resources_per_trial={"gpu": 1},  # 每个试验分配 1 个 GPU；如果你的机器有多 GPU，就能实现不同试验分别在不同卡上运行
        config=config,
        metric="accuracy",
        mode="max",
        progress_reporter=reporter,
        storage_path=model_save_path,  # 日志和检查点保存目录
        name="tune_experiment"
    )
    
    # 输出最佳超参数组合
    best_config = analysis.get_best_config(metric="accuracy", mode="max")

    with open("best_config","w",encode="utf-8") as f:
        f.write(str(best_config))
    print("Best config: ", best_config)


def main():
    # 解析参数，加载 tokenizer、数据集等
    args = args_parse()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    dataset = get_dataset(tokenizer, args)
    model_save_path = "./mt5-base"

    # 初始化 Ray（若 Ray 已经初始化，可忽略 ignore_reinit_error 参数）
    ray.init(ignore_reinit_error=True)

    # 启动 Ray Tune 超参数搜索
    tune_hyperparameters_ray(tokenizer, dataset, args, model_save_path)


if __name__ == '__main__':
    main()