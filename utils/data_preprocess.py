import os
import random
from collections import defaultdict
from typing import Union, List

from datasets import load_dataset
from torch.utils.data import Dataset
from datasets.dataset_dict import DatasetDict

from utils.ExtraNameSpace import DatasetsReaderNameSpace, DatasetsProcessorNameSpace
from utils.operators_concepts import operator_dict
from utils.text_utils import add_space_after_chinese, find_long_string_in_list


class PreliminaryExample:
    def __init__(self, expression, natural_sentence):
        self.expression = expression
        self.natural_sentence = natural_sentence

        self.__dict__.update({"expression": expression, "natural_sentence": natural_sentence})

    def __repr__(self):
        return f"Expression: {self.expression}\nNatural Sentence: {self.natural_sentence}"

class PreliminaryDataset(Dataset): # 这个类虽然名字叫了个预实验，但本身是指自建数据。因为自建数据没想到啥好名字
    def __init__(self, dataset: List[PreliminaryExample]):
        super().__init__()
        self.examples = dataset if dataset else [] # 用默认参数的话，会导致所有实例共享同一个list

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.examples[i]
        elif isinstance(i, slice):
            return PreliminaryDataset(self.examples[i])

        raise ValueError("Invalid index type {}.".format(type(i)))

    def __setitem__(self, key, value):
        if isinstance(key, slice) or isinstance(key, int):
            self.examples[key] = value
        else:
            raise ValueError("Invalid index type {}.".format(type(key)))

    def __delitem__(self, key):
        if isinstance(key, slice) or isinstance(key, int):
            del self.examples[key]
        else:
            raise ValueError("Invalid index type {}.".format(type(key)))

    def __len__(self):
        return len(self.examples)

    def append(self, expression, natural_sentence):
        self.examples.append(PreliminaryExample(expression, natural_sentence))

    def map(self, func, *args, **kwargs):
        return PreliminaryDataset([func(e, *args, **kwargs) for e in self])

    def filter(self, func, *args, **kwargs):
        return PreliminaryDataset([e for e in self if func(e, *args, **kwargs)])

    def shuffle(self):
        random.shuffle(self.examples)


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


@DatasetsReaderNameSpace.register("ours")
def read_dataset(directory_path: str) -> Union[Dataset, DatasetDict]:
    """
    :param directory_path: 读入目录内所有的文件
    :return:
    本函数指代常规的训练、测试集的那些读入
    """
    dataset = []

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, encoding='utf8') as f:
                line = eval(f.readlines()[0])

                for e in line:
                    expression, natural_sentence = e['表达式'], random.choice(e['自然语句'])
                    if expression == "None":
                        continue

                    dataset.append(PreliminaryExample(expression, natural_sentence))

    return PreliminaryDataset(dataset)

@DatasetsReaderNameSpace.register("topv2")
def read_unlabeled_dataset(directory_path: str):
    dataset = load_dataset(directory_path)
    return dataset["eval"]

@DatasetsReaderNameSpace.register("topv2")
def read_dataset(directory_path: str) -> Dataset:
    dataset = load_dataset(directory_path)
    return dataset

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

def unify_format(example: PreliminaryExample):
    replace_map = {"(": "（", ")": "）", ",": "，"}
    for key, value in replace_map.items():
        example.expression = example.expression.replace(key, value)
        example.natural_sentence = example.natural_sentence.replace(key, value)

    return example

@DatasetsReaderNameSpace.register("ours")
def ptr_change(example: PreliminaryExample):
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

            rhs_ptr = f"[{operator_dict[predicate][-1]}：" + "".join(
                ["@ptr_" + str(item + 1) for item in rhs_indexes]) + "]"
            variable_list = []
            # print("variables", variables)
            new_structural_tokens = []
            new_structural_tokens.append(f"[{operator_dict[predicate][-1]}：")
            for concept, variable in zip(operator_dict[predicate], variables):
                variable_indexes = find_long_string_in_list(word_list, variable)
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
    return e

@DatasetsReaderNameSpace.register("topv2")
def ptr_change(examples):
    """
    将semantic_parse里面的的词，换成utterance里对应的ptr_x
    """
    for i, st in enumerate(examples["semantic_parse"]):
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

        examples["semantic_parse"][i] = ' '.join(changed_item)
    return examples

def filter_function(example):
    return example.expression.find("：]") == -1

def preprocess_dataset(dataset):
    dataset = dataset.map(ptr_change)
    dataset = dataset.filter(filter_function)

    return dataset