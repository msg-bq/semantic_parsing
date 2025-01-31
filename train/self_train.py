from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import heapq

from torch import device
from tqdm import tqdm
from transformers import TrainingArguments, Trainer

from utils.dataset import mycollate_trainer, self_train_collate, AssertionExample, SelfTrainDataset

import torch.nn.functional as F
import pandas as pd
import os
# from FlagEmbedding import FlagReranker
import json

# with open("/home/lzx2000/test/Recall/yb_label.json","r",encoding="utf-8") as f:
#     data_dict = json.load(f)
# reranker = FlagReranker('/data/pretrained_models/bge-reranker-large', use_fp16=True)
#
# def get_list(inputs):
#
#     dict1 = {}
#
#     for key1 in data.keys():
#         scores = reranker.compute_score([inputs, key1])
#         dict1[key1] = scores[0]
#
#     all_top_keys = sorted(dict1, key=dict1.get, reverse=True)
#
#
#     top_keys = all_top_keys[:35]
#     top_scores = {key: dict1[key] for key in top_keys}
#
#     result = []
#
#     for key in top_scores.keys():
#         if top_scores[key] >= -4:
#             result.append(key)
#
#     return result
#
# def get_fw(sz_list,tokenizer):
#     # 使用 tokenizer 的 encode 方法将每个字符串转换成对应的 token
#     # 前面需要加上 \"谓词：
#     # 后面需要加上 \"
#     new_list = [f"\"谓词：{item}\"" for item in sz_list]
#
#     tokenized_list = [tokenizer.encode(text) for text in new_list]
#     return tokenized_list

class SelfTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        self.selftrain_iteration = kwargs.pop("selftrain_iteration", 3)  # self train的epoch
        self.selftrain_topk = kwargs.pop("selftrain_topk", 4)  # 这个值用于决定每轮获取软标注的topk
        self.max_length = kwargs.pop("max_length", 128)
        super().__init__(*args, **kwargs)

# class CustomLoss(nn.Module):
#     def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
#         super(CustomLoss, self).__init__()
#         # 定义你的损失函数所需的参数
# 
#     def forward(self, prediction, target):
#         # 定义损失函数的计算方式
#         # prediction 是模型的预测结果
#         # target 是真实标签
#         loss = ...  # 根据需要计算损失值的逻辑
#         return loss


class SelfTrainTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def softmax_chunks(self, lst, chunk_size=4):
        """Apply softmax to chunks of size chunk_size in lst."""
        # Convert list to numpy array
        arr = np.array(lst)
        # Reshape the array to have chunks of size chunk_size
        chunked_arr = arr.reshape(-1, chunk_size)
        # Apply softmax to each chunk
        softmaxed_chunks = np.apply_along_axis(F.softmax, 1, chunked_arr)
        # Flatten the result back to a 1D array if desired
        return softmaxed_chunks.flatten() if chunk_size == 1 else softmaxed_chunks

    def pad_tensor(self,tensor, target_length=127, padding_value=1.0):
        padding = 1 / self.args.selftrain_topk
        device = tensor.device  # 获取输入张量所在的设备
        k, current_length = tensor.size()
        if current_length >= target_length:
            return tensor
        else:
            padding_size = target_length - current_length
            padding = torch.full((k, padding_size), padding_value,device=device)
            padded_tensor = torch.cat((tensor, padding), dim=1)
            return padded_tensor

    def get_beam_search_score(self, question) -> dict:
        input_ids = self._tokenize_input_robustly(question)
        input_ids = input_ids.to(self.args.device)
        self.model.to(self.args.device)
        outputs = self.model.generate(input_ids=input_ids, num_beams=self.args.selftrain_topk, max_length=self.args.max_length,
                    num_return_sequences=self.args.selftrain_topk, return_dict_in_generate=True, output_scores=True)
        sequences = outputs.sequences  # [:, input_ids.shape[-1]:]
        # 这边得到k个outputs.score后，在每个token的维度上求softmax
        transition_scores = self.model.compute_transition_scores(
            # sequences, outputs_scores, outputs_beam_indices, normalize_logits=False
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        )
        # 在第0个维度上归一化

        transition_scores = F.softmax(transition_scores, dim=0)

        if outputs.sequences.shape[1] != len(outputs.scores):
            transition_scores = self.pad_tensor(transition_scores,self.args.max_length-1)
            ones = torch.ones(transition_scores.shape[0], 1).to(transition_scores.device)
            # # 将 ones 张量与原始 tensor 在维度 1 上进行拼接
            transition_scores = torch.cat((ones, transition_scores), dim=1)
        else:
            transition_scores = self.pad_tensor(transition_scores,self.args.max_length)
            
        # transition_scores = transition_scores.sum(dim=1).exp()
            
        sentence_score = {sent.cpu(): score.cpu() for sent, score in zip(sequences, transition_scores)} # 这里的transition_scores换成score列表
        return sentence_score

    # 拟定
    # def get_beam_search_score(self, question) -> dict:
    #     input_ids = self._tokenize_input_robustly(question)
    #     input_ids = input_ids.to(self.args.device)
    #     self.model.to(self.args.device)
    #
    #     # 先生成初步的输出
    #     outputs = self.model.generate(input_ids=input_ids,
    #                                   num_beams=self.args.selftrain_topk,
    #                                   max_length=self.args.max_length,
    #                                   num_return_sequences=self.args.selftrain_topk,
    #                                   return_dict_in_generate=True,
    #                                   output_scores=True)
    #
    #     sequences = outputs.sequences
    #     scores = outputs.scores  # 每个 token 的 logits 分数
    #
    #     # 1. 获取第一个生成的 token 的 logits
    #     first_token_logits = scores[0][0]  # 第一个生成的 token logits
    #
        # # 2. 定义需要限制的词表范围（离散的索引列表）
        # restricted_indices = [5, 15, 35, 25]  # 离散的词表索引
        # restricted_mask = torch.full_like(first_token_logits, -float('inf'))  # 设为负无穷
        # restricted_mask[restricted_indices] = first_token_logits[restricted_indices]  # 只保留在范围内的 logits
    #
    #     # 3. 获取在限制词表范围内的最大概率的 token
    #     restricted_logits = restricted_mask  # 限制范围后的 logits
    #     restricted_probs = F.softmax(restricted_logits, dim=-1)  # 转换为概率
    #     max_prob_token = torch.argmax(restricted_probs)  # 选择概率最大的 token
    #
    #     # 4. 用选择的 token 继续生成
    #     # 这里我们通过修改 logits 使得仅在 restricted_range 内的 token 能被选中
    #     next_input_ids = torch.cat([input_ids, max_prob_token.unsqueeze(0).unsqueeze(0)], dim=1)  # 拼接上选中的 token
    #
    #     # 继续生成后续的 tokens
    #     outputs = self.model.generate(input_ids=next_input_ids,
    #                                   num_beams=self.args.selftrain_topk,
    #                                   max_length=self.args.max_length,
    #                                   num_return_sequences=self.args.selftrain_topk,
    #                                   return_dict_in_generate=True,
    #                                   output_scores=True)
    #
    #     sequences = outputs.sequences
    #     transition_scores = self.model.compute_transition_scores(
    #         outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
    #     )
    #
    #     # 对 transition_scores 进行 softmax
    #     transition_scores = F.softmax(transition_scores, dim=0)
    #
    #     if outputs.sequences.shape[1] != len(outputs.scores):
    #         transition_scores = self.pad_tensor(transition_scores, self.args.max_length - 1)
    #         ones = torch.ones(transition_scores.shape[0], 1).to(transition_scores.device)
    #         transition_scores = torch.cat((ones, transition_scores), dim=1)
    #     else:
    #         transition_scores = self.pad_tensor(transition_scores, self.args.max_length)
    #
    #     sentence_score = {sent.cpu(): score.cpu() for sent, score in zip(sequences, transition_scores)}
    #     return sentence_score

    def _tokenize_input_robustly(self, input_text):
        """
        # 可以鲁棒地接纳tokenize前后的情况
        """
        tokenizer = self.tokenizer

        # 应该用args.max_length
        if isinstance(input_text, str):
            # 这里主要就是和之前的保持一致
            from utils.text_utils import add_space_after_chinese
            input_text = add_space_after_chinese(input_text.replace("得到", ""))
            from utils.tokenization import delete_blank
            tokenized_inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=self.args.max_length, return_tensors="pt")
            input_ids = delete_blank(tokenized_inputs)["input_ids"]
        elif isinstance(input_text, list):
            input_ids = torch.tensor(input_text)
        elif isinstance(input_text, torch.Tensor):
            input_ids = input_text
        else:
            raise ValueError("Invalid input_text type {}.".format(type(input_text)))

        return input_ids

    def get_fixed_output_score(self, example) -> float:
        input_ids = self._tokenize_input_robustly(example.natural_sentence)
        input_ids = input_ids.to(self.args.device)

        output_ids = self._tokenize_input_robustly(example.expression)
        output_ids = output_ids[output_ids != self.tokenizer.pad_token_id]
        output_ids = output_ids.to(self.args.device)
        score = 1

        sums = 0
        # print(output_ids)
        for ids in output_ids:
            
            tmp_output_ids = torch.cat((input_ids[0], ids.unsqueeze(0)), dim=0).unsqueeze(0) # [356,  481,  307, 1498,  284]
            # print("tmp_output_ids")
            # print(tmp_output_ids)
            # 上面的列表后面再拼个3，然后转成2维
            tmp_output = self.model.generate(input_ids=input_ids, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            # print("tmp_output")

            # 这里是发现不论是生成出来的token，还是ids，对应的score都是大于0的，而transition_scores反而是<0的
            # print(tmp_output.scores[0][0,ids])
            # print(print(tmp_output.scores[0][0,tmp_output.sequences[0][1]]))

            transition_scores = self.model.compute_transition_scores(
                tmp_output_ids, tmp_output.scores, normalize_logits=True
            )
            # print(transition_scores.item())
            sums += transition_scores.item()

            # score *= transition_scores.exp()

            input_ids = tmp_output_ids

        # print(f"sums:{sums}")
        # print(f"score:{score}")
        return score
        
    def get_soft_label_dataloader(self):
        # beam search
        self.train_dataset.topk = self.args.selftrain_topk
        dataset = self.train_dataset
        for question in tqdm(dataset.key_to_index.keys(), desc="Processing questions"):
            # print(dataset.unlabeled_dataset)
            mapp = {}
            last_examples = len(dataset[dataset.key_to_index[question]])

            sentence_score = self.get_beam_search_score(question)
            for sentence, score in sentence_score.items():
                # print(score)
                sentence = sentence.unsqueeze(dim=0) # 为了保持输入的维度还是2维，不要被for循环减少
                mapp[sentence] = score
                # 如果不在里面，则用原本的score
                if (question, sentence) not in self.train_dataset: # 如果beam search和原来的重叠率高会有点浪费，但这小问题了
                    # 还要有个weight
                    weight = score.sum(dim=0)
                    self.train_dataset.append(question,sentence, score,weight)
                # 如果在里面，要.....


            # 更新暂时只能这样了
            new_examples = dataset[dataset.key_to_index[question]]
            for example in new_examples[:last_examples]:
                new_score = mapp.get(example.natural_sentence,None)
                if new_score != None:
                    alpha = 0.5 # 这里有一个被固定进代码的变量
                    new_weight = new_score.sum(dim=0)
                    example.weight = (1-alpha) * example.weight + alpha * new_weight
                    example.score = (1-alpha) * example.score + alpha * new_score
                else:
                    example.weight = 0.7 * example.weight
                    example.score = 0.7 * example.score

            # new_examples = dataset[dataset.key_to_index[question]]
            # sum_scores = sum([example.weight for example in dataset[dataset.key_to_index[question]]])

            # # last_examples 从0，1，2，3，4....
            # # print(new_examples)
            # # print(last_examples)
            # for example in new_examples:
            #     # print(example)
            #     new_score = self.get_fixed_output_score(example)
            #     alpha = 0.5 # 这里有一个被固定进代码的变量
            #     example.weight = (1-alpha) * example.weight / sum_scores + alpha * new_score
            # # print("2222")
            # # 归一化
            # sum_scores = sum([example.weight for example in new_examples])

            # for example in new_examples:
            #     example.weight /= sum_scores
                # print(example.weight)
        

    # def calc_weighted_loss(loss_fct, outputs, batch, scores):
    #     inputs, labels = batch["input_ids"], batch["labels"]
    #     loss_per_token = loss_fct(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1))

    #     # 由于每个样例的长度不同，需要计算每个样例的平均loss
    #     loss_per_token = loss_per_token.view(inputs.size(0), -1)
    #     loss_per_example = loss_per_token.sum(dim=1) / (labels != tokenizer.pad_token_id).sum(dim=1)

    #     whole_loss = sum([l * s for l, s in zip(loss_per_example, scores)]) / len(scores)
    #     # 打印每个样例的loss
    #     #     for i, loss in enumerate(loss_per_example):
    #     # ImportError}: {loss.item()}")
    #     return whole_loss

    def compute_loss(self, model, inputs, return_outputs=False):  ## compute loss这个步骤实际上定义了 forward和loss的计算过程

        labels = inputs.get("labels")
        score = inputs.pop("score")
        inputs.pop("weight")
        # # 创建一个形状为 [8, 1] 的张量，所有元素为 1
        outputs = model(**inputs)
        is_a_tensor = torch.is_tensor(score)
        if is_a_tensor == False:
            for s in score:
                print(s.shape)
            score = torch.stack(score).to(labels.device)
        
        score = score.view(-1)

        logits = outputs.logits  # .get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)

        loss_fct = nn.CrossEntropyLoss(reduction='none',ignore_index=-100)

        print(loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).shape)
        loss = (score * loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) ).mean()

        # final_loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        # return (torch.masked_select(loss, labels.view(-1, 1) != -1).mean(), outputs) if return_outputs else loss
        return ( loss, {'outputs': outputs}) if return_outputs else loss  # 一定要记得 compute loss的时候！！！outputs要用字典返回
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predicted = logits.argmax(-1)
    correct_predictions = torch.all(torch.eq(predicted, labels), dim=1)
    lens = labels.size(0)
    return {"accuracy": correct_predictions/lens}

def occupy_gpu_memory(gpu_ids, size_in_gb):
    """占用指定GPU的显存."""
    occupied_memory = {}
    for gpu_id in gpu_ids:
        device = torch.device(f"cuda:{gpu_id}")
        # 创建占位张量，占用指定大小的显存
        dummy_tensor = torch.empty(int(size_in_gb * 1024**3 // 4), dtype=torch.float32, device=device)
        occupied_memory[gpu_id] = dummy_tensor
        print(f"Occupied {size_in_gb} GB on GPU {gpu_id}.")
    return occupied_memory



def train_model_self_train(model, tokenizer, optimizer, dataset, args):
    """
    1. 先训练基础模型
    2. 用基础模型预测unlabeled数据集
    3. 选择topk数据集
    4. 用topk数据集fine-tune基础模型
    5. 重复2-4
    """
    from transformers import AutoTokenizer, AdamW, get_scheduler

    train_args = TrainingArguments(output_dir=args.save_dir,
                                   num_train_epochs=50,#args.epoch,  # 这个指每个self_train里面的epoch
                                   per_device_train_batch_size=args.batch_size*args.selftrain_topk,
                                   save_steps=1000,
                                   save_total_limit=1,
                                   learning_rate=args.lr,
                                   evaluation_strategy="epoch",
                                   do_eval=True if "eval" in dataset else False,
                                   no_cuda=False if args.device == "cuda" else True)

    unlabeled_dataset = dataset["unlabeled"]  # 我们假设这里是个question_list
    # unlabeled_dataset = SelfTrainDataset(question_list=unlabeled_dataset)
    selftrain_args = SelfTrainingArguments(output_dir=args.save_dir,
                                           num_train_epochs=1,  # 这个指每个self_train里面的epoch
                                           per_device_train_batch_size=args.batch_size,
                                           save_steps=1000,
                                           save_total_limit=1,
                                           learning_rate=args.lr,
                                           evaluation_strategy="no",
                                           selftrain_topk=args.selftrain_topk,
                                           max_length=128,
                                           dataloader_pin_memory=False,
                                           do_eval=False,
                                           no_cuda=False if args.device == "cuda" else True)

    for epoch in range(args.epoch):

        if args.given_model: # 如果基础模型已经训过了，就先训self
            args.given_model = False
        else:
            # 1. 先训练基础模型
            
            trainer = Trainer(model=model,
                              args=train_args,
                              data_collator=mycollate_trainer, # 要么给自己的，要么在定义trainer后面单独写一个data_collator=None，不然代码里有默认collate
                              train_dataset=dataset["train"],
                              eval_dataset=dataset["train"],#dataset["eval"] if "eval" in dataset else None,
                              tokenizer=tokenizer,
                              optimizers=(optimizer, None)) # 缺了学习率调度器

            trainer.train()
            model = trainer.model
            model.to(args.device)

        print("初步训练完成")
        # exit()
        # 2. 用基础模型预测unlabeled数据集
        # self_trainer = SelfTrainTrainer(model=model,
        #                                 args=selftrain_args,
        #                                 data_collator=self_train_collate,
        #                                 train_dataset=unlabeled_dataset,
        #                                 tokenizer=tokenizer,
        #                                 optimizers=(optimizer, None))


        # unlabeled_dataset.eval()
        # self_trainer.get_soft_label_dataloader()

        # unlabeled_dataset.train(tokenizer, args.max_length)
        # self_trainer.train()
        
        # model = self_trainer.model

        model.save_pretrained(f"/home/lzx2000/temp/lzx/lzx/test/test/semantic_parsing_few_shot_3/mt5-base/mt5_our_data")