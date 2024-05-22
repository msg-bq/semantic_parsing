import torch
import torch.nn as nn
import heapq

from transformers import TrainingArguments, Trainer

from utils.dataset import mycollate


class SelfTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selftrain_iteration = kwargs.get("selftrain_iteration", 3) # self train的epoch
        self.selftrain_topk = # 这个值用于决定每轮获取软标注的topk

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
        self.unlabeled_dataset = kwargs.get("unlabeled_dataset")
        self.args = kwargs.get("args")
        self.trainer = kwargs.get("trainer") # 我们把正常的trainer传进来，这样就不用重复写train了

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

    def get_soft_label_dataloader(self):
        # beamsearch
        device = self.trainer.device
        model = self.trainer.model
        selftrain_topk = self.args.selftrain_topk
        unlabeled_dataset = self.unlabeled_dataset

        soft_labeled_dataset = []

        for i, batch in enumerate(unlabeled_dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.generate(batch['input_ids'], num_beams=selftrain_topk, max_length=512,
                                 num_return_sequences=selftrain_topk, return_dict_in_generate=True, output_scores=True)
            sequences = out.sequences # logits区别

            top_k_scores = self.softmax_chunks(out.sequences_scores)

            soft_labeled_dataset.append({"input": })

    def train(self):
        super(SelfTrainTrainer, self).train()

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
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, collate_fn=mycollate)

    model.eval()
    with torch.no_grad():
        for batch in unlabeled_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask)
            lm_logits = outputs.logits
            # 3. 选择topk数据集
            # 4. 用topk数据集fine-tune基础模型
            # 5. 重复2-4
            # TODO
            break


