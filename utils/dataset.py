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
        if isinstance(other, AssertionExample):
            return self.expression == other.expression and self.natural_sentence == other.natural_sentence

        raise ValueError("Invalid AssertionExample comparison.")
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
    def __init__(self, init_question_list: str, topk=5):#unlabeled_dataset: List[List[AssertionExample]]=None, topk=5):
        """
        注：一般情况下，question存的是str，natural_sentence存的是tensor


        这个地方用堆的复杂度仍然是nlogn，主要瓶颈是每轮存在一次整个数组重新打分的过程
        O(n)的策略也有，就每次遍历前转list再重新建堆，但这个就太没必要了，且数量少那常数项大也不合适
        heapq.heappush(items, new_item)
        max_item = heapq.heappop(items)

        综上所述，暂定还是list
        """
        super().__init__()
        self.key_to_index = {q: i for i, q in enumerate(init_question_list)}
        # for i, data_list in enumerate(self.unlabeled_dataset):  # 这里list没关系，因为每次都是更新所有的score，所以每次整个把data_list删掉重建
        #     # 因为即便用dict，修改方便但每次还要排序
        #     key = data_list[0].expression
        #     self.key_to_index[key] = i

        self.unlabeled_dataset = [[] for _ in range(len(init_question_list))]
        self.sent_to_instance_list = [] # 用于避免重复
        for i, data_list in enumerate(self.unlabeled_dataset):
            # sent_to_instance = {data.natural_sentence: data for data in data_list}# 这个地方就不应该有重复
            sent_to_instance = {}
            self.sent_to_instance_list.append(sent_to_instance)

        self.sorted_sign = [False] * len(init_question_list) # 用于减少排序开销

        self.tokenized_dataset = None
        self.tokenized_sign = False

        self.topk = topk

    def __getitem__eval(self, i):
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

    def __getitem__train(self, i):
        return self.tokenized_dataset[i]

    def __getitem__(self, i):
        """
        只取topk
        """
        # 这里面没考虑不够topk的情况，这个一般不会触发
        if self.tokenized_sign:
            return self.__getitem__train(i)
        
        return self.__getitem__eval(i)


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
            self.unlabeled_dataset[self.key_to_index[key]].append(AssertionExample(expression, natural_sentence, score)) #?
            self.sorted_sign[self.key_to_index[key]] = False

        else:
            self.key_to_index[key] = len(self.unlabeled_dataset)
            self.unlabeled_dataset.append([AssertionExample(expression, natural_sentence, score)])
            self.sorted_sign.append(False)

    def train(self, tokenizer, max_length=512):
        """进入train的状态，此时应该改为返回tokenize_dataset"""
        self.tokenized_dataset = self._return_tokenized_dataset(tokenizer, max_length) # 每次重新算吧，毕竟labels在更新，也浪费不了多少时间
        self.tokenized_sign = True # 顺序不能颠倒，不然getitem会错

    def eval(self):
        """进入eval的状态，此时应该按正常返回，以便更新topk"""
        del self.tokenized_dataset
        self.tokenized_sign = False

    def _return_tokenized_dataset(self, tokenizer, max_length=512) -> List[List[Dict[str, torch.Tensor]]]:
        """
        这个地方直接返回可以进dataloader的dataset/list, 不过这里因为有topk，所以要多一层
        """
        def tokenize_example(input_text):
            if isinstance(input_text, str):
                input_ids =  tokenizer(input_text, padding='max_length', truncation=True, max_length=max_length,
                                       return_tensors="pt")["input_ids"]
            elif isinstance(input_text, list):
                input_ids = torch.tensor(input_text + [tokenizer.pad_token_id] * (max_length - len(input_text)))
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension
            elif isinstance(input_text, torch.Tensor):
                if input_text.size(0) < max_length:
                    padding = torch.tensor([tokenizer.pad_token_id] * (max_length - input_text.size(0)))
                    input_ids = torch.cat((input_text.cpu(), padding), dim=0)
            else:
                raise ValueError("Invalid input_text type {}.".format(type(input_text)))

            return input_ids.cpu() # 这个地方固定住to.cpu也没关系，因为目测没有to(cuda)的需求

        tokenized_dataset = []
        for topk_examples in self:
            tokenized_dataset.append(
                [{"input_ids": tokenize_example(example.natural_sentence),
                  "labels": tokenize_example(example.expression),
                  "weight": example.weight}
                for example in topk_examples]) # 因为getitem时候已经取了topk

        return tokenized_dataset

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

def self_train_collate(examples):
    """
    和mycollate一样，主要是每四个要单独处理一个y^
    """
    for topk_examples in examples:
        weights_sum = sum([example['weight'] for example in topk_examples])

        print("examples", examples)
        for topk_examples in examples:
            print(topk_examples)
            for sub_example in topk_examples: # 指topk
                for key in sub_example:
                    if key == "weight":
                        sub_example[key] /= weights_sum

                    try:
                        sub_example[key] = torch.tensor(sub_example[key])
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
