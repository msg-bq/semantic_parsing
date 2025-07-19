import os
import warnings

import sys
from typing import Literal

sys.path.append('/usr/src/app/semantic_parsing/code')  # 按理说不需要，但服务器有短暂的时间出现索引错误，稳妥起见补一个ad hoc的处理

import yaml

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import get_scheduler

import re
import string
# import jsonlines
from tqdm import tqdm

def format_time_string(input_string):
    english_punctuation = string.punctuation.replace(':', '').replace("'", '')
    time_pattern = r'(\d{1,2})(:)?(\d{2})?(am|pm|AM|PM)?'
    def replacer(match):
        hour = match.group(1)
        colon = " : " if match.group(2) else ""
        minute = match.group(3) if match.group(3) else ""
        period = f" {match.group(4)}" if match.group(4) else ""
        return f"{hour}{colon}{minute}{period}"
    formatted_string = re.sub(time_pattern, replacer, input_string)

    def insert_spaces_around_punctuation(formatted_string):
        i = 0
        while i < len(formatted_string):
            if formatted_string[i] in english_punctuation:
                if i > 0 and formatted_string[i - 1] != " ":
                    formatted_string = formatted_string[:i] + " " + formatted_string[i:]
                    i += 1  # 跳过新插入的空格
                if (i + 1) < len(formatted_string) and formatted_string[i + 1] != " ":
                    formatted_string = formatted_string[:i + 1] + " " + formatted_string[i + 1:]
                    i += 1  # 跳过新插入的空格
            i += 1
        return formatted_string

    formatted_string = insert_spaces_around_punctuation(formatted_string)
    return re.sub(r"([a-zA-Z0-9])'s", r"\1 's", formatted_string)


import re
def in_input(content, input_str):
    if f" {content} " not in input_str:
        if f" {content}?" not in input_str and f" {content}." not in input_str and f" {content}," not in input_str:
            if f" {content}" not in input_str:
                return False
            elif input_str.endswith(f" {content}") == False:
                return False

    return True

def edit_label(examples):
    input = examples["utterance"]
    output = examples["semantic_parse"]

    pattern = r"(?<=\[SL:WEATHER_TEMPERATURE_UNIT\s)(.*?)(?=\s\])"
    if "[SL:WEATHER_TEMPERATURE_UNIT" in output:
        match = re.search(pattern, output)
        content = match.group(0)
        if in_input(content, input) == False:
            in_input1 = content.capitalize()
            if in_input(in_input1, input) == True:
                output = re.sub(pattern, in_input1, output)
            elif in_input(content.lower(), input) == True:
                output = re.sub(pattern, content.lower(), output)
            else:
                return examples

    examples["semantic_parse"] = output

    return examples

from utils.remove_non_slot_leaf import remove_non_slot_leaf_nodes
from utils.sort_label import sort_string

def ptr_change(examples):
    examples["semantic_parse"] = edit_label(examples)["semantic_parse"]

    examples["utterance"] = format_time_string(examples["utterance"])
    # 删标签
    examples["semantic_parse"] = remove_non_slot_leaf_nodes(examples["semantic_parse"])
    # 排序
    examples["semantic_parse"] = sort_string(examples["semantic_parse"])
    # print(examples)
    return examples


def preprocess_dataset(dataset):
    dataset = dataset.map(ptr_change, load_from_cache_file=False)
    return dataset


def tokenize_function(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=64,
                                 return_tensors="pt")
    tokenized_labels = tokenizer(examples['semantic_parse'], padding='max_length', truncation=True, max_length=64,
                                 return_tensors="pt")

    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    return tokenized_inputs


def tokenizer_dataset(tokenizer, dataset):
    tokenized_datasets = dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer}, load_from_cache_file=False)
    tokenized_datasets = tokenized_datasets.remove_columns(["utterance"])
    tokenized_datasets = tokenized_datasets.remove_columns(["semantic_parse"])
    if "domain" in tokenized_datasets.column_names:
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

def extract_slots(label):
    # 按空格切分字符串
    parts = label.split()[1:-1]
    result = []  # 用来存储最外层的SL标签
    stack = []  # 用来处理嵌套情况
    current_slot = ""  # 当前的SL标签
    for part in parts:
        if part.startswith('[sl:'):  # 如果是SL标签
            if current_slot == "":  # 如果sl为空，表示是最外层的SL标签
                current_slot = part  # 开始新的最外层SL标签
            else:  # 如果栈不为空，表示在处理嵌套的SL标签
                current_slot += ' ' + part  # 拼接当前的SL标签
            stack.append('[')  # 入栈表示有一个嵌套
        elif part.startswith('[in:'):
            current_slot += ' ' + part
            stack.append('[')
        elif part == ']' or part.find("]") != -1:  # 处理结束的右括号
            try:
                stack.pop()  # 出栈
                current_slot += ' ' + part  # 拼接右括号
                if not stack:  # 栈为空，表示这个 ] 就是当前最外层的slot
                    result.append(current_slot)  # 将最外层的SL标签存入结果
                    current_slot = ""  # 重置当前SL标签
            except:
                continue
        else:
            current_slot += ' ' + part  # 拼接标签内容
    # 返回最终结果
    return result

def get_f1(label, predict):
    predict_slot_lst = extract_slots(predict.lower())
    label_lst = extract_slots(label.lower())

    true_index = []
    # 算出来正确的有几个
    true_predict = 0
    for predict_slot in predict_slot_lst:
        predict_fix = predict_slot.split()[0]
        predict_content = predict_slot.split()[1:-1]
        predict_content_str = " ".join(predict_content)
        for l, label_slot in enumerate(label_lst):
            if predict_fix not in label_slot or l in true_index:
                continue
            label_content = label_slot.split()[1:-1]
            label_content_str = " ".join(label_content)
            if predict_content_str.replace(" ","") in label_content_str.replace(" ","") or label_content_str.replace(" ","") in predict_content_str.replace(" ",""):
                true_predict += 1
                true_index.append(l)

    result = {
        "true_predict": true_predict,
        "predict_slot_lst_len": len(predict_slot_lst),
        "label_lst_len": len(label_lst),
    }

    return result

def remove_prepositions(text):
    prepositions = ["in", "on", "at", "by", "with", "about","the",
                     "for", "under", "over", "between", "during",
                     "through", "of"]
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in prepositions]
    return " ".join(filtered_words)


def test_model(model, tokenizer, dataset, args=None, device=None, train_machine: Literal['our', 'ray'] = 'our'):
    # 在 GPU 上测试（如果可用）
    if args != None:
        device = args.device
    correct = 0
    data_length = 0

    true_predict = 0
    predict_slot_lst_len = 0
    label_lst_len = 0
    # DataLoader 用于批量测试
    # template = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    # for data in dataset["validation"]:
    #     print(data)
    #     data["semantic_parse"] = template.format(content=data["semantic_parse"])

    if train_machine == 'our':
        for key in dataset:
            dataset[key] = tokenizer_dataset(tokenizer, preprocess_dataset(dataset[key]))

    test_loader = DataLoader(dataset["validation"], batch_size=128, collate_fn=mycollate_trainer)  # 你可以调整 batch_size

    for i, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            generate_ids = model.generate(batch['input_ids'], max_length=128)

            input_ids = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            decoded_predicted = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            k = 0

            num_correct_sentences = 0
            for input_id, pred, label in zip(input_ids, decoded_predicted, decoded_labels):
                k += 1
                result = get_f1(label, pred)

                pred = remove_prepositions(pred).lower()
                label = remove_prepositions(label.lower())

                true_predict += result["true_predict"]
                predict_slot_lst_len += result["predict_slot_lst_len"]
                label_lst_len += result["label_lst_len"]

                pred = pred[pred.index('['):] if '[' in pred else pred
                print("结果：", pred, '分隔符', label)

                pred = pred.replace(" ", "").strip()
                pred = _extract_pred(pred)
                label = label.replace(" ", "").strip()

                if pred == label:
                    num_correct_sentences += 1

            data_length += k
            correct += num_correct_sentences

    accuracy = correct / data_length
    acc = true_predict / predict_slot_lst_len if predict_slot_lst_len > 0 else 0
    recall = true_predict / label_lst_len if label_lst_len > 0 else 0
    f1_score = 2 * acc * recall / (acc + recall) if (acc + recall) > 0 else 0
    print(f"f1-score:{f1_score}")
    return accuracy, f1_score


# def _load_dataset(path):
#     dataset = load_dataset(path)
#     for key in dataset:
#         dataset[key] = tokenizer_dataset(tokenizer, preprocess_dataset(dataset[key]))
#
#     return dataset


def _extract_pred(s: str) -> str:
    s = s.strip()
    if not s:
        return s

    if s[0] != '[':
        warnings.warn("s[0] must be a [")
        return s

    result = ''
    depth = 0

    for i, c in enumerate(s):
        result += c
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return result

    return result


def test_all_models(model_output_dir, model_name, save_path, **kwargs) -> list[str]:
    tasks = ['event', 'reminder', 'weather']
    dataset_types = ['top', 'our_data', 'our_add_top']
    exp_settings = ['SPIS25', 'SPIS50', 'full']
    train_strategy = 'full'

    preprocessed_data_save_dir = '../LLaMA-Factory/data'

    results = []

    for task in tasks:
        for dataset_type in dataset_types:
            for exp_setting in exp_settings:
                if dataset_type == 'our_data' and exp_setting == 'SPIS50':
                    file_name = f"{task}_{dataset_type}_{'SPIS25'}"
                else:
                    file_name = f"{task}_{dataset_type}_{exp_setting}"
                model_dir = f'{model_output_dir}/{model_name}/{train_strategy}/sft_{file_name}'

                # try:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
                model = AutoModelForCausalLM.from_pretrained(model_dir).cuda()

                data_path = os.path.join(preprocessed_data_save_dir, task, dataset_type, exp_setting)
                dataset = load_dataset(data_path)  # fixme: 这里缺少处理

                acc, f1 = test_model(model, tokenizer, dataset, device='cuda')

                del model
                # except Exception as e:
                #     print("测试错误", model_dir, e)
                #     acc, f1 = -1, -1

                result = f"{task} {dataset_type} {exp_setting} {model_name}: acc is {acc} and f1 is {f1}"

                with open(result_save_path, 'a', encoding='utf-8') as f:
                    f.write(result + '\n')

                results.append(result)


    return results


if __name__ == '__main__':
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # model_name = 'qwen3_8b'
    # model_path = f'/usr/src/app/saves/{model_name}/full/sft_event_top_SPIS25'
    #
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    #
    # data_path = '/home/cyz/data/semantic_parsing/LLaMA-Factory/data/event/top/SPIS25'
    # dataset = load_dataset(data_path)
    # acc, f1 = test_model(model, tokenizer, dataset, device='cuda')
    # print(acc, f1)

    machine_path = '/home/cyz/data/semantic_parsing/semantic_parsing'  # '/usr/src/app'
    saved_models_dir = '/var/lib/docker/data/cyz/semantic_parsing'  # '/usr/src/app'

    model_info = {
        'qwen3_8b': {
            'model_output_dir': f'{saved_models_dir}/saves',
            'model_name': 'qwen3_8b',
        },
        'llama3_8b': {
            'model_output_dir': f'{saved_models_dir}/saves',
            'model_name': 'llama3_8b',
        },
        'Phi_4': {
            'model_output_dir': f'{saved_models_dir}/saves',
            'model_name': 'Phi_4',
        }
    }

    result_save_path = f'{machine_path}/saves/results.txt'
    with open(result_save_path, "w", encoding="utf-8") as f:
        pass

    for key in model_info:
        test_all_models(save_path=result_save_path, **model_info[key])

