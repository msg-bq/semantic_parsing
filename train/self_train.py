from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import heapq

from torch import device
from transformers import TrainingArguments, Trainer

from utils.dataset import mycollate, self_train_collate, AssertionExample

import torch.nn.functional as F



class SelfTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selftrain_iteration = kwargs.get("selftrain_iteration", 3) # self train的epoch
        self.selftrain_topk = kwargs.get("selftrain_topk", 4)# 这个值用于决定每轮获取软标注的topk

class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(CustomLoss, self).__init__()
        # 定义你的损失函数所需的参数

    def forward(self, prediction, target):
        # 定义损失函数的计算方式
        # prediction 是模型的预测结果
        # target 是真实标签
        loss = ...  # 根据需要计算损失值的逻辑
        return loss


class SelfTrainTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlabeled_dataset = kwargs.get("unlabeled_dataset") # 这里不需要dataloader，只要一个dataset就行
        self.args = kwargs.get("args")

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

    def get_beam_search_score(self, example) -> dict:
        input_ids = self.tokenizer(example.question, return_tensors="pt").input_ids \
            if isinstance(example.question, str) else example.question # 可以接收tokenize前后的情况

        outputs = self.model.generate(input_ids=input_ids, num_beams=self.args.selftrain_topk, max_length=512,
                    num_return_sequences=self.args.selftrain_topk, return_dict_in_generate=True, output_scores=True)

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        )

        transition_scores = transition_scores.sum(dim=1).exp()
        sequences = outputs.sequences[:, input_ids.shape[-1]:]
        sentence_score = {sent: score for sent, score in zip(sequences, transition_scores)} # 每个sent是ids，即[356,  481,  307, 1498,  284]

        return sentence_score

    def get_fixed_output_score(self, example, natural_sentence) -> float:
        input_ids = self.tokenizer(example.question, return_tensors="pt").input_ids \
            if isinstance(example.question, str) else example.question

        output_ids = self.tokenizer(natural_sentence, return_tensors="pt").input_ids \
            if isinstance(natural_sentence, str) else natural_sentence

        score = 1

        for ids in output_ids:
            tmp_output_ids = torch.cat((input_ids, ids.unsqueeze(0)), dim=1).unsqueeze(0) # [356,  481,  307, 1498,  284]
            # 后面再拼个3，然后转成2维

            tmp_output = self.model.generate(input_ids=input_ids, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            transition_scores = self.model.compute_transition_scores(
                tmp_output_ids, tmp_output.scores, normalize_logits=True
            )

            score *= transition_scores.exp()

            input_ids = tmp_output_ids.squeeze(0)

        return score

    def get_soft_label_dataloader(self):
        # beam search
        device = self.device
        model = self.model
        selftrain_topk = self.args.selftrain_topk
        unlabeled_dataset = self.unlabeled_dataset

        soft_labeled_dataset = []

        for example in unlabeled_dataset:
            self.get_beam_search_score(example)

    def compute_loss(self, model, inputs, return_outputs=False):  ## compute loss这个步骤实际上定义了 forward和loss的计算过程
        labels = inputs.get("labels")
        outputs = model(inputs.get('inputs'))  ##在这里定义了foward和batch的计算过程
        logits = outputs  # .get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")  ## loss可以不断重新定义无所谓的
        loss = loss_fct(logits.view(-1, 1), labels.float().view(-1, 1))
        # final_loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        # return (torch.masked_select(loss, labels.view(-1, 1) != -1).mean(), outputs) if return_outputs else loss
        return (
        loss, {'outputs': outputs}) if return_outputs else loss  # 一定要记得 compute loss的时候！！！outputs要用字典返回


def train_model_self_train(model, optimizer, dataset, args):
    """
    1. 先训练基础模型
    2. 用基础模型预测unlabeled数据集
    3. 选择topk数据集
    4. 用topk数据集fine-tune基础模型
    5. 重复2-4
    """

    # 1. 先训练基础模型
    train_args = TrainingArguments(output_dir=args.save_dir,
                                   num_train_epochs=args.epoch, # 这个指每个self_train里面的epoch
                                   per_device_train_batch_size=args.batch_size,
                                   save_steps=1000,
                                   learning_rate=args.lr,
                                   evaluation_strategy="epoch")

    trainer = Trainer(model=model,
                      args=train_args,
                      data_collator=mycollate, # 要么给自己的，要么在定义trainer后面单独写一个data_collator=None，不然代码里有默认collate
                      train_dataset=dataset["train"],
                      eval_dataset=dataset["eval"],
                      tokenizer=model.tokenizer,
                      optimizers=(optimizer, None)) # 缺了学习率调度器

    trainer.train()


    # 2. 用基础模型预测unlabeled数据集
    unlabeled_dataset = dataset["unlabeled"]
    # unlabeled_dataset 还是没有标签的

    def give_x_y_score():
        pass

    # unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, collate_fn=self_train_collate)




