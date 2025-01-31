import argparse
from collections import defaultdict
from typing import Union, Tuple

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"

import json
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

from utils.data_preprocess import read_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.ExtraNameSpace import NameSpace
# import higher
from label_utils import label_correct_sentence


def args_parse():
    parser = argparse.ArgumentParser(description="semantic_parser")

    parser.add_argument('--config', type=str, default='config.yaml',
                    help='config file, 只使用单层的嵌套')

    parser.add_argument("--dataset", type=str, default="topv2",
                        choices=["ours", "topv2", "zcl", "zcl_mixed"])

    parser.add_argument("--train_dataset_dir", type=str, default="/home/lzx2000/temp/lzx/lzx/test/test/semantic_parsing_few_shot/TOPv2/low_resource_splits/weather",
                    help="train dataset dir")

    parser.add_argument("--unlabel_dataset_dir", type=str, default="/home/lzx2000/temp/lzx/lzx/test/test/semantic_parsing_few_shot/TOPv2/low_resource_splits/weather",
                    help="unlabel dataset dir")

    parser.add_argument("--test_dataset_dir", type=str, default="./data/dev",
                        help="test dataset dir")

    parser.add_argument("--task", type=str, default="self-train", choices=["preliminary", "self-train"],
                    help="省得分文件了")

    parser.add_argument("--model_dir", type=str,
                        default="/home/lzx2000/temp/lzx/lzx/test/test/semantic_parsing5/mt5-base/mt5-train",#"/data/lbq/models/mt5-base-trained-final-500+500-2-7_again",
    #,"/home/lzx/T5-base/model3/mt5-base-trained-final-500+500-2-7_again"
                    help="model dir")

    parser.add_argument("--save_dir", type=str, default="./mt5-base",
                    help="save dir")

    parser.add_argument("--experiment_name", type=str, default="Default", choices=["Default"], # 这个比如10样例、100样例等
                    help="experiment name")

    parser.add_argument("--optimizer", type=str, default="AdamW",
                    help="optimizer")

    parser.add_argument("--lr", type=float, default=3e-5,
                    help="learning rate")

    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss",
                    help="criterion")

    parser.add_argument("--device", type=str, default="cuda",
                    help="device")

    parser.add_argument("--epoch", type=int, default=20,
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

    parser.add_argument("--selftrain_topk", type=int, default=4,
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


import re
import string
# 英文标点符号
def format_time_string(input_string):
    english_punctuation = string.punctuation.replace(':', '')
    # 正则表达式匹配时间格式
    time_pattern = r'(\d{1,2})(:)?(\d{2})?(am|pm|AM|PM)?'

    # 替换逻辑
    def replacer(match):
        hour = match.group(1)
        colon = " : " if match.group(2) else ""
        minute = match.group(3) if match.group(3) else ""
        period = f" {match.group(4)}" if match.group(4) else ""
        return f"{hour}{colon}{minute}{period}"

    # 对字符串进行替换
    formatted_string = re.sub(time_pattern, replacer, input_string)

    # 判断末尾字符是否是english_punctuation内的标点符号且前面无空格
    if formatted_string[-1] in english_punctuation and formatted_string[-2]!= " ":
        formatted_string = formatted_string[:-1] + " " + formatted_string[-1]

    return formatted_string

def ptr_change(examples):
    """
    将semantic_parse里面的的词，换成utterance里对应的ptr_x
    """
    # print(examples)
    st = examples["semantic_parse"]
    changed_item = []
    cnt = 1

    for s in st.split(' '):
        # 如果是以 [ 开头 或 是 ] 或 是英文标点符号，则保留原样
        if s.startswith('[') or s == ']':
            changed_item.append(s)
        else:
            changed_item.append(f"@ptr_{cnt}")
            cnt += 1

    examples["semantic_parse"] = ' '.join(changed_item)
    examples["utterance"] = format_time_string(examples["utterance"])
    # print(examples)
    return examples

def replace_ptr_with_words(sentence, ptr_string):
    """
    Replace @ptr_X in the second string with the corresponding word from the sentence.

    Parameters:
        sentence (str): The original sentence (e.g., "Set the alarm for every 2 hours staring in one hour.").
        ptr_string (str): The string with @ptr_X placeholders (e.g., "[IN:CREATE_ALARM @ptr_1 @ptr_2 ...]").

    Returns:
        str: The updated ptr_string with @ptr_X replaced by the corresponding words from the sentence.
    """
    # Split the sentence into words

    words = sentence.split()

    # Find all placeholders like @ptr_X in the ptr_string
    import re
    ptr_placeholders = re.findall(r"@ptr_\d+", ptr_string)

    # Replace each @ptr_X with the corresponding word
    replaced_string = ptr_string
    for placeholder in ptr_placeholders:
        # Extract the index from the placeholder (e.g., @ptr_3 -> 3)
        index = int(placeholder.split("_")[1]) - 1

        # Replace the placeholder with the word if the index is valid, else leave as is
        if index < len(words):
            replaced_string = replaced_string.replace(placeholder, words[index], 1)
        else:
            replaced_string = replaced_string.replace(placeholder, "<undefined>", 1)
    return replaced_string

def preprocess_dataset(dataset):
    dataset = dataset.map(ptr_change)
    # dataset = dataset.filter(filter_function)

    return dataset

def tokenize_function(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    tokenized_labels = tokenizer(examples['semantic_parse'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")

    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    return tokenized_inputs

def tokenizer_dataset(tokenizer, dataset):
    tokenized_datasets = dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer})
    tokenized_datasets = tokenized_datasets.remove_columns(["utterance"])
    tokenized_datasets = tokenized_datasets.remove_columns(["semantic_parse"])
    tokenized_datasets = tokenized_datasets.remove_columns(["domain"])

    return tokenized_datasets

def mycollate_trainer(examples):
    """
    这里面不应当包含多余的key
    """
    remove_keys = []
    for example in examples:
        for key in example:
            try:
                example[key] = torch.tensor(example[key])
            except Exception as e:
                remove_keys.append(key)

    for key in set(remove_keys):
        for example in examples:
            del example[key]

    batch = {}
    for key in examples[0]:
        batch[key] = torch.stack([example[key][0] for example in examples])

    return batch

tokenizer = AutoTokenizer.from_pretrained("/data/pretrained_models/mt5-base")

args = args_parse()

dataset = read_dataset("/home/lzx2000/temp/lzx/lzx/test/test/semantic_parsing_few_shot_3/TOPv2/low_resource_splits/our/data")

dataset["train"] = tokenizer_dataset(tokenizer, preprocess_dataset(dataset["train"]))

model = AutoModelForSeq2SeqLM.from_pretrained("/home/lzx2000/temp/lzx/lzx/test/test/semantic_parsing_few_shot_3/mt5-base/mt5_our_data")
# 在 GPU 上测试（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

correct = 0
gz_correct = 0
data_length = 0
# DataLoader 用于批量测试
test_loader = DataLoader(dataset["train"], batch_size=8, collate_fn=mycollate_trainer)  # 你可以调整 batch_size

file1 = open("decoded_predictions1.txt", "w", encoding="utf-8")
for i, batch in enumerate(test_loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

        logits = outputs.logits
        # 使用 argmax 获取每个样本的预测类别
        predicted = logits.argmax(-1)
        # 检查整个句子是否完全匹配
        correct_predictions = torch.all(torch.eq(predicted, batch["labels"]), dim=1)
        # 解码 predicted 和 labels
        input_ids = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        decoded_predicted = tokenizer.batch_decode(predicted, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        print(decoded_predicted)
        # 将它们组合成一行行写入文件，格式为 "预测值\t标签值"
        # k = 0
        #
        # gz = 0
        # for input_id, pred, label in zip(input_ids, decoded_predicted, decoded_labels):
        #     # 把pred 和 label还原成里面的词
        #     pred_nt = replace_ptr_with_words(input_id, pred)
        #     pred_nw = label_correct_sentence(input_id, pred_nt)
        #     # 用label_correct_sentence把predict做一次处理看看
        #     label_nt = replace_ptr_with_words(input_id, label)
        #
        #     if pred_nw == label_nt:
        #         gz  += 1
        #
        #     file1.write(f"excample {k}:\n{input_id}\norigin_label:\n{label}\nlabel:\n{label_nt}\npred_修复前:\n{pred_nt}\npred_修复后:\n{pred_nw}\n")
        #     file1.flush()  # 手动刷新缓冲区，确保数据写入磁盘
        #
        #     k += 1

        # 获取准确的句子数量
        num_correct_sentences = correct_predictions.sum().item()

        correct += num_correct_sentences
        # gz_correct += gz
        # 获取句子的数量
        num_sentences = batch["labels"].size(0)

        data_length = data_length + num_sentences
        print(f" \r第{i}个batch Accuracy: {num_correct_sentences / num_sentences}")
        # print(f" \r第{i}个batch Gz_Accuracy: {gz / num_sentences}")

# 计算最终准确率
print(f"\rAccuracy: {correct / data_length}\n")
# print(f"\rGz_Accuracy: {gz_correctcorrect / data_length}\n")