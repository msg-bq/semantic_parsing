import os
import random
from collections import defaultdict
from typing import Union, List, Dict

from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from datasets.dataset_dict import DatasetDict

from utils.ExtraNameSpace import DatasetsReaderNameSpace, DatasetsProcessorNameSpace
from utils.dataset import AssertionExample, PreliminaryDataset, SelfTrainDataset
from utils.operators_concepts import operator_dict
from utils.text_utils import add_space_after_chinese, find_long_string_in_list

import pandas as pd
import json


@DatasetsProcessorNameSpace.register("Default")
def split_dataset(dataset: Union[list, Dataset, DatasetDict], split_ratio: Union[float, list]) -> tuple:
    """
    :param dataset: 数据集
    :param split_ratio: 划分比例，如果是列表，要求必须够3个
    :return: 划分后的数据集
    """
    if isinstance(dataset, DatasetDict): # 这个的话，就默认已经是拆分过的数据了
        # 先假设key是train, eval/dev, test。如果还有其他的，就先认了
        return dataset["train"], dataset["eval"] if "eval" in dataset else dataset["dev"], dataset["test"]

    random.shuffle(dataset)

    if isinstance(split_ratio, float):
        split_index = int(len(dataset) * split_ratio)
        return dataset[:split_index], None, dataset[split_index:]
    elif isinstance(split_ratio, list):
        assert len(split_ratio) == 3
        split_index = [int(len(dataset) * ratio) for ratio in split_ratio]
        return dataset[:split_index[0]], dataset[split_index[0]:split_index[1]], dataset[split_index[1]:]
    else:
        raise ValueError("Invalid split ratio type.")


def read_ours_from_dir(directory_path: str) -> List[AssertionExample]:
    dataset = []

    s = 0
    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, encoding='utf8') as f:
                line = eval(f.readlines()[0])

                for e in line:
                    # 存在e['自然语句'] = []的情况导致random.choice(e['自然语句'])报错
                    if(e['自然语句'] == []):
                        continue
                    expression, natural_sentence = e['表达式'], random.choice(e['自然语句'])
                    # if expression == "None":
                    #     continue
                    dataset.append(AssertionExample(expression, natural_sentence))
    return dataset

@DatasetsReaderNameSpace.register("preliminary_ours")
def read_dataset(directory_path: str) -> Union[Dataset, DatasetDict]:
    """
    :param directory_path: 读入目录内所有的文件
    :return:
    本函数指代常规的训练、测试集的那些读入
    """
    dataset = read_ours_from_dir(directory_path)

    return PreliminaryDataset(dataset)

from torch.utils.data import random_split
@DatasetsReaderNameSpace.register("self-train_ours")
def read_dataset(directory_path: str) -> Union[Dict, DatasetDict]:
    """
    一般情况下，read的时候就改成train eval test
    """
    train_dataset = read_ours_from_dir(directory_path)

    # 计算训练集和开发集的大小
    # train_size = int(0.95 * len(train_dataset))
    # dev_size = len(train_dataset) - train_size

    # # 使用random_split划分数据集
    # train_subset, dev_subset = random_split(train_dataset, [train_size, dev_size])
    return {'train': PreliminaryDataset(train_dataset)}

@DatasetsReaderNameSpace.register("self-train_topv2")
def read_dataset(directory_path: str) -> DatasetDict:
    """
    一般情况下，read的时候就改成train eval test
    """
    dataset = load_dataset(directory_path)
    train_dataset = dataset["train"]
    return dataset

@DatasetsReaderNameSpace.register("self-train_zcl")
def read_dataset(directory_path: str) -> DatasetDict:
    dataset1000 = load_dataset("/data/lbq/datasets/1000spis", split='train')
    # dataset1000 = load_dataset("/home/lzx/old_selftrain/data/weather/1000spis", split='train')
    dataset = dataset1000.train_test_split(test_size=0.5)
    return {"train": dataset["train"]}

@DatasetsReaderNameSpace.register("self-train_zcl_mixed")
def read_dataset(directory_path: str) -> DatasetDict:
    def filter_function(example):
        return example["seqlogical"].find("[IN:UNSUPPORTED_WEATHER") != -1

    dataset = load_dataset("/data/lbq/datasets/1000spis", split="train")
    dataset1 = dataset.filter(filter_function)
    dataset1 = dataset1.train_test_split(test_size=0.5)

    # 再度入一个TopV2的，取0.2
    def filter_function(example):
        return example["seqlogical"].find("[IN:UNSUPPORTED_WEATHER") == -1

    dataset2 = dataset.filter(filter_function)
    dataset2 = dataset2.train_test_split(test_size=0.8)

    # 最后读入自建的
    with open("/data/lbq/datasets/train_weather.json", 'r') as file:
        dataset3 = json.load(file)
    data_dict = {
        "seqlogical": [item["expression"] for item in dataset3],
        "utterance": [item["sentence"] for item in dataset3],
    }

    dataset3 = Dataset.from_dict(data_dict)

    combined_train = concatenate_datasets(
        [dataset1['train'], dataset2['train'], dataset3])

    return {"train": combined_train}

@DatasetsReaderNameSpace.register("ours")
def read_unlabeled_dataset(directory_path: str):
    # 修改为 ↓
    for filename in os.listdir(directory_path):
        # 读取json文件，里面有"split_and_filter"和"origin"
        # 把split_and_filter里的全部取出来就行
        with open(directory_path + "/" + filename) as f: # replace 'yourfilename' with your actual file name
            data = json.load(f)
        flattened_list = []
        for i in range(len(data)):
            flattened_list += data[i]["split_and_filter"]
    return SelfTrainDataset(init_question_list=flattened_list)

@DatasetsReaderNameSpace.register("topv2")
def read_unlabeled_dataset(directory_path: str):
    # dataset = load_dataset(directory_path)
    # 用于存储所有字典数据的列表
    query_list = []
    with open(directory_path, "r", encoding="utf-8") as f:
        for s in f:
            dict1 = json.loads(json.loads(s))
            query_list.append(dict1["question"])
    return SelfTrainDataset(init_question_list=query_list)
    # return dataset["eval"]["utterance"]

# zcl
def read_unlabeled_dataset_zcl(directory_path: str):
    dataset = load_dataset("/data/lbq/datasets/temp", split='train') #/home/lzx/old_selftrain/data/temp

    dataset3 = load_dataset("/data/lbq/datasets/delelte", split='train') #   /home/lzx/old_selftrain/data/weather/delelte
    df1 = pd.DataFrame(dataset3)
    df2 = pd.DataFrame(dataset)

    # 找出df2中与df1重复的部分
    df2_unique = df2[~df2['utterance'].isin(df1['utterance'])]
    df2_unique = df2_unique.sample(2000, random_state=42)

    # 如果需要，将处理后的数据转换回Dataset格式
    dataset = Dataset.from_pandas(df2_unique)

    return SelfTrainDataset(init_question_list=dataset["utterance"][:2])

@DatasetsReaderNameSpace.register("zcl")
def read_unlabeled_dataset(directory_path: str):
    return read_unlabeled_dataset_zcl(directory_path)

@DatasetsReaderNameSpace.register("zcl_mixed")
def read_unlabeled_dataset(directory_path: str):
    return read_unlabeled_dataset_zcl(directory_path)

def select_dataset(dataset: Dataset, args) -> dict:
    """
    从数据集中筛选一定量算子和一定量的数据，纯粹为了实验的平衡
    """
    data_by_label = defaultdict(list)
    for data in dataset:
        expression = data.expression
        first_operator = expression.split("(")[0].strip()
        data_by_label[first_operator].append(data)

    if args.seed:
        random.seed(args.seed)

    selected_operators = random.sample(list(data_by_label.keys()), min(args.operator_num, len(data_by_label)))\
        if args.operator_num else list(data_by_label.keys())

    selected_dataset = {}
    for operator in selected_operators:
        example_num = min(args.example_num, len(data_by_label[operator]))
        selected_dataset[operator] = random.sample(data_by_label[operator], example_num)

    if isinstance(dataset, PreliminaryDataset):
        selected_dataset = {k: PreliminaryDataset(v) for k, v in selected_dataset.items()}

    return selected_dataset

def unify_format(example: AssertionExample):
    replace_map = {"(": "（", ")": "）", ",": "，"}
    for key, value in replace_map.items():
        example.expression = example.expression.replace(key, value)
        example.natural_sentence = example.natural_sentence.replace(key, value)

    return example

@DatasetsProcessorNameSpace.register("ours")
def ptr_change(example: AssertionExample):

    e = unify_format(example)

    e.natural_sentence = add_space_after_chinese(e.natural_sentence.replace("得到", ""))
    # encode = tokenizer.encode(e.natural_sentence)
    word_list = e.natural_sentence.split()
    if e.expression != "None":
        result_list = []
        expression_list = e.expression.split(" ，")
        for expression in expression_list:
            word_list = e.natural_sentence.split()
            # for enc in encode:
            #     word_list.append(tokenizer.decode(enc))

            st = expression
            nl = e.natural_sentence
            st = st.replace(' ', '')
            lhs, rhs = st.strip().split('=')
            assert lhs.strip().endswith('）')

            import re

            text = lhs.strip()[:-1]
            result = text.split('（')

            predicate = result[0]
            predicate = predicate.replace("得到", "")
            predicate1 = f"\"谓词：{predicate}\""
            variables = result[1]
            # predicate, variables = lhs.strip()[:-1].

            variables = variables.split('，')

            rhs_indexes = find_long_string_in_list(word_list, rhs)

            # 改成留着第一个和最后一个，看看能不能好点
            if len(rhs_indexes) > 0:
                rhs_indexes = [rhs_indexes[0],rhs_indexes[-1]]

            rhs_ptr = f"[{operator_dict[predicate][-1]}：" + "".join(
                ["@ptr_" + str(item + 1) for item in rhs_indexes]) + "]"
            variable_list = []
            # print("variables", variables)
            new_structural_tokens = []
            new_structural_tokens.append(f"[{operator_dict[predicate][-1]}：")
            for concept, variable in zip(operator_dict[predicate], variables):
                variable_indexes = find_long_string_in_list(word_list, variable)
                if len(variable_indexes) > 0:
                    variable_indexes = [variable_indexes[0],variable_indexes[-1]]
                result = "".join(["@ptr_" + str(item + 1) for item in variable_indexes])
                variable_list.append(f"[{concept}：{result}]")
                new_structural_tokens.append(f"[{concept}：")

            combined_list = ','.join(variable_list)

            # 往词表中加入词
            # tokenizer.add_tokens(list(set(new_structural_tokens)))

            result = f'{predicate1}({combined_list})={rhs_ptr}'
            result_list.append(result)
        Result = ""
        for i, result in enumerate(result_list):
            Result += result
            if i < len(result_list) - 1:
                Result += " , "
        e.expression = Result
    # print(e)
    return e

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

@DatasetsProcessorNameSpace.register("topv2")
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

#zcl
def ptr_change_zcl(examples):
    """
    将semantic_parse里面的的词，换成utterance里对应的ptr_x
    """
    st = examples["seqlogical"]
    changed_item = []
    # ut_list = ut.split(' ')
    cnt = 1
    # print(st)
    for s in st.split(' '):
        if s.startswith('[') or s == ']':
            # print(s)
            # exit()
            changed_item.append(s)
        else:
            # print(s)
            # print(f"@ptr_{cnt}")
            changed_item.append(f"@ptr_{cnt}")
            cnt += 1

    examples["seqlogical"] = ' '.join(changed_item)
    return examples
@DatasetsProcessorNameSpace.register("zcl")
def ptr_change(examples):
    return ptr_change_zcl(examples)

@DatasetsProcessorNameSpace.register("zcl_mixed")
def ptr_change(examples):
    return ptr_change_zcl(examples)

@DatasetsProcessorNameSpace.register("Default")
def filter_function(example):
    return example.expression.find("：]") == -1

@DatasetsProcessorNameSpace.register("zcl") # weather
def filter_function(example):
    return example
@DatasetsProcessorNameSpace.register("zcl_mixed")
def filter_function(example):
    return example

def preprocess_dataset(dataset):
    dataset = dataset.map(ptr_change)
    # dataset = dataset.filter(filter_function)

    return dataset

# if __name__ == '__main__':
#     s = "[IN:GET_ALARM Check my alarms . ]"
#     import string
#     english_punctuation = string.punctuation.replace(':', '')
#
#     def ptr_change(examples):
#         """
#         将semantic_parse里面的的词，换成utterance里对应的ptr_x
#         """
#         # print(examples)
#         st = examples["semantic_parse"]
#         changed_item = []
#         cnt = 1
#
#         for s in st.split(' '):
#             # 如果是以 [ 开头 或 是 ] 或 是英文标点符号，则保留原样
#             if s.startswith('[') or s == ']' or all(c in english_punctuation for c in s):
#                 changed_item.append(s)
#             else:
#                 changed_item.append(f"@ptr_{cnt}")
#                 cnt += 1
#
#         examples["semantic_parse"] = ' '.join(changed_item)
#         examples["utterance"] = format_time_string(examples["utterance"])
#         # print(examples)
#         return examples
#
#     print(ptr_change({"semantic_parse": s, "utterance": s}))
