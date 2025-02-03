import argparse
from collections import defaultdict
from typing import Union, Tuple

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



def test_model(model, tokenizer, dataset, args):
    # 在 GPU 上测试（如果可用）
    device = args.device
    correct = 0
    data_length = 0
    total_loss = 0
    # DataLoader 用于批量测试
    test_loader = DataLoader(dataset["validation"], batch_size=8, collate_fn=mycollate_trainer)  # 你可以调整 batch_size
    for i, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            # 使用 argmax 获取每个样本的预测类别
            predicted = logits.argmax(-1)
            # 检查整个句子是否完全匹配
            correct_predictions = torch.all(torch.eq(predicted, batch["labels"]), dim=1)
            num_correct_sentences = correct_predictions.sum().item()
            correct += num_correct_sentences
            # gz_correct += gz
            # 获取句子的数量
            num_sentences = batch["labels"].size(0)
            data_length = data_length + num_sentences

            # 计算交叉熵损失
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            total_loss += loss.item()
    # 计算最终准确率
    accuracy = correct / data_length
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss