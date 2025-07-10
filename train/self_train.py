from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import heapq

from torch import device
from transformers import TrainingArguments, Trainer

from utils.dataset import mycollate_trainer, self_train_collate, AssertionExample, SelfTrainDataset

import torch.nn.functional as F



class SelfTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        self.selftrain_iteration = kwargs.pop("selftrain_iteration", 3)  # self train的epoch
        self.selftrain_topk = kwargs.pop("selftrain_topk", 4)  # 这个值用于决定每轮获取软标注的topk
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
        self.train_args = kwargs.pop("train_args")
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

    def get_beam_search_score(self, question) -> dict:
        input_ids = self._tokenize_input_robustly(question)
        input_ids = input_ids.to(self.train_args.device)
        self.model.to(self.train_args.device)

        outputs = self.model.generate(input_ids=input_ids, num_beams=self.train_args.selftrain_topk, max_length=512,
                    num_return_sequences=self.train_args.selftrain_topk, return_dict_in_generate=True, output_scores=True)

        sequences = outputs.sequences  # [:, input_ids.shape[-1]:]
        # outputs_scores = tuple([s.to("cpu") for s in outputs.scores])
        # outputs_beam_indices = outputs.beam_indices.to("cpu")

        transition_scores = self.model.compute_transition_scores(
            # sequences, outputs_scores, outputs_beam_indices, normalize_logits=False
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        )

        transition_scores = transition_scores.sum(dim=1).exp()

        if outputs.sequences[0].shape[-1] >= input_ids.shape[-1]:
            # print("The output sequence is not longer than the input sequence.")
            sequences = sequences[:, input_ids.shape[-1]:] #outputs.

        sentence_score = {sent.cpu(): score.cpu() for sent, score in zip(sequences, transition_scores)} # 每个sent是ids，即[356,  481,  307, 1498,  284]

        return sentence_score

    def _tokenize_input_robustly(self, input_text):
        """
        # 可以鲁棒地接纳tokenize前后的情况
        """
        tokenizer = self.tokenizer

        if isinstance(input_text, str):
            input_ids = tokenizer(input_text, padding='max_length', truncation=True, max_length=30,
                                  return_tensors="pt")["input_ids"]
        elif isinstance(input_text, list):
            input_ids = torch.tensor(input_text)
        elif isinstance(input_text, torch.Tensor):
            input_ids = input_text
        else:
            raise ValueError("Invalid input_text type {}.".format(type(input_text)))

        return input_ids

    def get_fixed_output_score(self, example) -> float:
        input_ids = self._tokenize_input_robustly(example.natural_sentence)
        input_ids = input_ids.to(self.train_args.device)

        output_ids = self._tokenize_input_robustly(example.expression)
        output_ids = output_ids[output_ids != self.tokenizer.pad_token_id]

        score = 1

        for ids in output_ids:
            tmp_output_ids = torch.cat((input_ids[0], ids.unsqueeze(0)), dim=0).unsqueeze(0) # [356,  481,  307, 1498,  284]
            # 上面的列表后面再拼个3，然后转成2维

            tmp_output = self.model.generate(input_ids=input_ids, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            transition_scores = self.model.compute_transition_scores(
                tmp_output_ids, tmp_output.scores, normalize_logits=True
            )

            score *= transition_scores.exp()

            input_ids = tmp_output_ids

        return score

    def get_soft_label_dataloader(self):
        # beam search
        self.train_dataset.topk = self.train_args.selftrain_topk
        dataset = self.train_dataset

        for question in dataset.key_to_index.keys():
            last_examples = len(dataset.unlabeled_dataset[dataset.key_to_index[question]])

            sentence_score = self.get_beam_search_score(question)
            for sentence, score in sentence_score.items():
                sentence = sentence.unsqueeze(dim=0) # 为了保持输入的维度还是2维，不要被for循环减少
                if (question, sentence) not in self.train_dataset: # 如果beam search和原来的重叠率高会有点浪费，但这小问题了
                    self.train_dataset.append(question, sentence, score)

            new_examples = dataset[dataset.key_to_index[question]]
            sum_scores = sum([example.weight for example in dataset[dataset.key_to_index[question]]])

            for example in new_examples[:last_examples]:
                new_score = self.get_fixed_output_score(example)
                alpha = 0.5 # 这里有一个被固定进代码的变量
                example.weight = (1-alpha) * example.weight / sum_scores + alpha * new_score

            # 归一化
            sum_scores = sum([example.weight for example in new_examples])

            for example in new_examples:
                example.weight /= sum_scores

    def calc_weighted_loss(loss_fct, outputs, batch, scores):
        inputs, labels = batch["input_ids"], batch["labels"]
        loss_per_token = loss_fct(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1))

        # 由于每个样例的长度不同，需要计算每个样例的平均loss
        loss_per_token = loss_per_token.view(inputs.size(0), -1)
        loss_per_example = loss_per_token.sum(dim=1) / (labels != tokenizer.pad_token_id).sum(dim=1)

        whole_loss = sum([l * s for l, s in zip(loss_per_example, scores)]) / len(scores)
        # 打印每个样例的loss
        #     for i, loss in enumerate(loss_per_example):
        # ImportError}: {loss.item()}")
        return whole_loss

    def compute_loss(self, model, inputs, return_outputs=False):  ## compute loss这个步骤实际上定义了 forward和loss的计算过程
        score = torch.tensor(inputs.pop("weight"))
        return torch.sum(score * super(SelfTrainTrainer, self).compute_loss(model, inputs, return_outputs))
        # labels = inputs.get("labels")
        # outputs = model(inputs.get('inputs'))  ##在这里定义了foward和batch的计算过程
        # logits = outputs  # .get("logits")
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.BCEWithLogitsLoss(reduction="mean")  ## loss可以不断重新定义无所谓的
        # loss = loss_fct(logits.view(-1, 1), labels.float().view(-1, 1))
        # # final_loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        # # return (torch.masked_select(loss, labels.view(-1, 1) != -1).mean(), outputs) if return_outputs else loss
        # return (
        # loss, {'outputs': outputs}) if return_outputs else loss  # 一定要记得 compute loss的时候！！！outputs要用字典返回


from tqdm import tqdm
import torch.nn.functional as F
def get_unlabel_data(unlabel_train_loader, model,tokenizer, save_path: str, domain:str, p_rate: float):
    
    # 用于暂存所有样本及其置信度
    all_data = []

    # 遍历无标签数据的 DataLoader
    sum = 0
    true = 0
    for batch in tqdm(unlabel_train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            # 关键：需要打开 return_dict_in_generate 和 output_scores
            outputs = model.generate(
                batch['input_ids'], 
                max_length=128,
                return_dict_in_generate=True,
                output_scores=True,
            )
            # 使用模型的 compute_transition_scores 方法计算转移分数
            transition_scores = model.copmute_transition_scores(
                outputs.sequences, outputs.scores, beam_indices=None, normalize_logits=False
            )
            # 在第 0 个维度上归一化
            # transition_scores = F.softmax(transition_scores, dim=0)
            confidence_scores = transition_scores.sum(dim=1)
            # 计算每条生成序列的置信度
            # outputs.scores 是一个列表，长度等于生成的 token 步数
            # 每个元素 shape = [batch_size, vocab_size]
            batch_size = batch['input_ids'].shape[0]
            for b_idx in range(batch_size):
                # 解码输入序列和生成的序列（可根据需要 skip_special_tokens 等参数）
                input_text = tokenizer.decode(batch["input_ids"][b_idx], skip_special_tokens=True)
                pred_text = tokenizer.decode(outputs.sequences[b_idx], skip_special_tokens=True)
                conf = confidence_scores[b_idx].item()

                data_dict = {
                    "question": input_text,
                    "pred": pred_text,
                    "confidence": conf
                }
                if unlabel_filter(data_dict,domain):
                    all_data.append(data_dict)
    
    # 排序并取置信度处于前 30% 的数据
    all_data = sorted(all_data, key=lambda x: x["confidence"], reverse=True)
    top_n = int(len(all_data) * p_rate)
    top_data = all_data[:top_n]

    true0 = 0
    for item in top_data:
        if item['pred'] == item['label']:
            true0 += 1
    if true0 / len(top_data) > 0.85:
        p_rate += 0.1
        if p_rate >= 1:
            p_rate = 1
    
    p_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for p1 in p_lst:
        top_n1 = int(len(all_data) * p1)
        test_data = all_data[:top_n1]
        # 看每個區間下面的分數
        true1 = 0
        for item in test_data:
            if item['pred'] == item['label']:
                true1 += 1
        file_any.write(f"{p1}的比例下：\t{true1}/{len(test_data)}\n")
        file_any.flush()
        
        
    # 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in top_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    return p_rate


def get_selftrain_model(tokenizer, unlabel_path: str):
    train_data = []
    with jsonlines.open(unlabel_path) as reader:
        for line in reader:
            source_text = line['question']
            target_text = line['pred']
            # 对输入文本进行编码
            source_encoding = tokenizer(source_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
            input_ids = source_encoding["input_ids"]
            attention_mask = source_encoding["attention_mask"]

            # 对目标文本进行编码
            target_encoding = tokenizer(target_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
            labels = target_encoding["input_ids"]

            train_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    return train_data


def train_model_self_train(model, tokenizer, optimizer, dataset, args):
    """
    1. 先训练基础模型
    2. 用基础模型预测unlabeled数据集
    3. 选择topk数据集
    4. 用topk数据集fine-tune基础模型
    5. 重复2-4
    """
    train_args = TrainingArguments(output_dir=args.save_dir,
                                   num_train_epochs=1,#args.epoch,  # 这个指每个self_train里面的epoch
                                   per_device_train_batch_size=args.batch_size,
                                   save_steps=1000,
                                   learning_rate=args.lr,
                                   evaluation_strategy="epoch",
                                   do_eval=True if "eval" in dataset else False,
                                   no_cuda=False if args.device == "cuda" else True)

    unlabeled_dataset = dataset["unlabeled"]  # 我们假设这里是个question_list
    # unlabeled_dataset = SelfTrainDataset(question_list=unlabeled_dataset)

    selftrain_args = SelfTrainingArguments(output_dir=args.save_dir,
                                           num_train_epochs=args.selftrain_iteration,  # 这个指每个self_train里面的epoch
                                           per_device_train_batch_size=args.batch_size,
                                           save_steps=1000,
                                           learning_rate=args.lr,
                                           evaluation_strategy="epoch",
                                           selftrain_topk=2,
                                           no_cuda=False if args.device == "cuda" else True)

    # model.resize_token_embeddings(len(tokenizer))

    for epoch in range(3):
        if args.given_model: # 如果基础模型已经训过了，就先训self
            args.given_model = False
        else:
            # 1. 先训练基础模型
            trainer = Trainer(model=model,
                              args=train_args,
                              data_collator=mycollate_trainer, # 要么给自己的，要么在定义trainer后面单独写一个data_collator=None，不然代码里有默认collate
                              train_dataset=dataset["train"][:10],
                              eval_dataset=dataset["train"][:10],#dataset["eval"] if "eval" in dataset else None,
                              tokenizer=tokenizer,
                              optimizers=(optimizer, None)) # 缺了学习率调度器

            trainer.train()
            model = trainer.model
            model.to(args.device)

            print("初步训练完成")


            # 自训练
            unlabel_train_loader = DataLoader(unlabeled_dataset, batch_size=128, collate_fn=mycollate_trainer)
            p_rate = 0.2
            unlabel_path = ""
            # 得到数据
            p_rate = get_unlabel_data(unlabel_train_loader, model,tokenizer, unlabel_path,"weather",p_rate)
            # 自训练
            unlabel_train_dataset = get_selftrain_model(tokenizer, unlabel_path)
            selftrain_args = TrainingArguments(
                output_dir=f"{args.save_dir}/normal",
                num_train_epochs=25,
                per_device_train_batch_size=128,
                learning_rate=args.lr,
                do_eval=False,
                no_cuda=False
            )

            # 定义Trainer实例
            selftrainer = Trainer(
                model=model,
                args=selftrain_args,
                data_collator=mycollate_trainer,
                train_dataset=unlabel_train_dataset
            )
            # 开始自训练
            selftrainer.train()

            model.save_pretrained(f"/data/lbq/models/mt5_1000_{epoch}")