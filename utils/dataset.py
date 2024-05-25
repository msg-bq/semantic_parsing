import random
from collections import defaultdict
from typing import List, Dict

import torch
from torch.utils.data import Dataset


class AssertionExample:
    def __init__(self, expression, natural_sentence, weight=-1):
        self.expression = expression
        self.natural_sentence = natural_sentence
        self.weight = weight # 只是self train时候才用

        self.__dict__.update({"expression": expression,
                              "natural_sentence": natural_sentence,
                              "weight": weight})

    def __repr__(self):
        return f"Expression: {self.expression}\nNatural Sentence: {self.natural_sentence}"

    def __lt__(self, other):
        return self.weight < other.weight

    def __hash__(self):
        return hash((self.expression, self.natural_sentence))

    def __eq__(self, other):
        return self.expression == other.expression and self.natural_sentence == other.natural_sentence

class PreliminaryDataset(Dataset): # 这个类虽然名字叫了个预实验，但本身是指自建数据。因为自建数据没想到啥好名字
    def __init__(self, dataset: List[AssertionExample]=None):
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
        self.examples.append(AssertionExample(expression, natural_sentence))

    def map(self, func, *args, **kwargs):
        return PreliminaryDataset([func(e, *args, **kwargs) for e in self])

    def filter(self, func, *args, **kwargs):
        return PreliminaryDataset([e for e in self if func(e, *args, **kwargs)])

    def shuffle(self):
        random.shuffle(self.examples)


class SelfTrainDataset(Dataset):
    def __init__(self, question_list: str, topk=5):#unlabeled_dataset: List[List[AssertionExample]]=None, topk=5):
        """
        这个地方用堆的复杂度仍然是nlogn，主要瓶颈是每轮存在一次整个数组重新打分的过程
        O(n)的策略也有，就每次遍历前转list再重新建堆，但这个就太没必要了，且数量少那常数项大也不合适
        heapq.heappush(items, new_item)
        max_item = heapq.heappop(items)

        综上所述，暂定还是list
        """
        super().__init__()
        self.key_to_index = {q: i for i, q in enumerate(question_list)}
        # for i, data_list in enumerate(self.unlabeled_dataset):  # 这里list没关系，因为每次都是更新所有的score，所以每次整个把data_list删掉重建
        #     # 因为即便用dict，修改方便但每次还要排序
        #     key = data_list[0].expression
        #     self.key_to_index[key] = i

        self.unlabeled_dataset = [[]] * len(question_list)
        self.sent_to_instance_list = [] # 用于避免重复
        for i, data_list in enumerate(self.unlabeled_dataset):
            sent_to_instance = {data.natural_sentence: data for data in data_list}# 这个地方就不应该有重复
            self.sent_to_instance_list.append(sent_to_instance)

        self.sorted_sign = [False] * len(question_list) # 用于减少排序开销

        self.topk = topk

    def __getitem__(self, i):
        """
        只取topk
        """
        # 这里面没考虑不够topk的情况，这个一般不会触发

        if isinstance(i, int):
            if not self.sorted_sign[i]:
                self.unlabeled_dataset[i].sort(key=lambda x: x.weight, reverse=True)
                self.sorted_sign[i] = True

            return self.unlabeled_dataset[i][:self.topk]
        elif isinstance(i, slice):
            for j in range(i.start, i.stop):
                if not self.sorted_sign[j]:
                    self.unlabeled_dataset[j].sort(key=lambda x: x.weight, reverse=True)
                    self.sorted_sign[j] = True

            return [self.unlabeled_dataset[j][:self.topk] for j in range(i.start, i.stop)]

        raise ValueError("Invalid index type {}.".format(type(i)))

    def __len__(self):
        return len(self.unlabeled_dataset)

    def __contains__(self, item):
        expression, natural_sentence = item

        sent_to_instance = self.sent_to_instance_list[self.key_to_index[expression]]
        if natural_sentence in sent_to_instance:
            return True

        return False


    def append(self, expression, natural_sentence, score=-1):
        """
        判断新旧再append，这里面不控制
        """
        key = expression
        if key in self.key_to_index:
            self.unlabeled_dataset[self.key_to_index[key]].append(AssertionExample(expression, natural_sentence, score))
            self.sorted_sign[self.key_to_index[key]] = False
        else:
            self.key_to_index[key] = len(self.unlabeled_dataset)
            self.unlabeled_dataset.append([AssertionExample(expression, natural_sentence, score)])
            self.sorted_sign.append(False)


def mycollate(examples):
    for example in examples:
        for key in example:
            try:
                example[key] = torch.tensor(example[key])
            except Exception as e:
                pass

    batch = {}
    for key in examples[0]:
        try:
            batch[key] = torch.stack([example[key] for example in examples])
        except:
            batch[key] = [example[key] for example in examples]

    return batch

def self_train_collate(examples):
    """
    和mycollate一样，主要是每四个要单独处理一个y^
    """
    for topk_examples in examples:
        weights_sum = sum([example.weight for example in topk_examples])

        for example in examples:
            for key in example:
                if key == "weight":
                    example[key] /= weights_sum

                try:
                    example[key] = torch.tensor(example[key])
                except Exception as e:
                    pass

    examples = [e for topk_examples in examples for e in topk_examples]

    batch = {}
    for key in examples[0]:
        try:
            batch[key] = torch.stack([example[key] for example in examples])
        except:
            batch[key] = [example[key] for example in examples]

    return batch
