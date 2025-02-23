import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import get_scheduler

import re
import string
import jsonlines
from transformers import RobertaModel, RobertaTokenizer
from module.RoBERTa import RobertaBiLinearAttentionModel


# 英文标点符号
def format_time_string(input_string: str) -> str:
    english_punctuation = string.punctuation.replace(':', '')
    # 正则表达式匹配时间格式
    time_pattern = r'(\d{1,2})(:)?(\d{2})?(:)?(\d{2})?(am|pm|AM|PM)?'
    # 替换逻辑
    def replacer(match):
        hour = match.group(1)
        colon = " : " if match.group(2) else ""
        minute = match.group(3) if match.group(3) else ""
        colon_2 = " : " if match.group(4) else ""
        second = match.group(5) if match.group(5) else ""
        period = f" {match.group(6)}" if match.group(6) else ""
        return f"{hour}{colon}{minute}{colon_2}{second}{period}"
    # 对字符串进行替换
    formatted_string = re.sub(time_pattern, replacer, input_string)
    # 判断末尾字符是否是english_punctuation内的标点符号且前面无空格
    if formatted_string[-1] in english_punctuation and formatted_string[-2] != " ":
        formatted_string = formatted_string[:-1] + " " + formatted_string[-1]
    # 使用正则表达式找到字母后面的单引号，并在其前面添加一个空格
    return re.sub(r"([a-zA-Z])'s", r"\1 's", formatted_string)

from utils.remove_non_slot_leaf import remove_non_slot_leaf_nodes
from utils.sort_label import sort_string

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
        content = match.group(0).replace(" ","")
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

def filter(examples):
    input = examples["utterance"]
    output = examples["semantic_parse"]

    # 过滤掉today误生成的
    if input.find("today") != -1 and output.find("today") == -1:
        return False
    if input.find("Today") != -1 and output.find("Today") == -1:
        return False

    # 判断[SL:WEATHER_TEMPERATURE_UNIT的内容在label里有没有
    pattern = r"(?<=\[SL:WEATHER_TEMPERATURE_UNIT\s)(.*?)(?=\s\])"
    if "[SL:WEATHER_TEMPERATURE_UNIT" in output:
        match = re.search(pattern, output)
        content = match.group(0).replace(" ", "")
        return in_input(content, input)

    return True

def ptr_change(examples):
    """
    将semantic_parse里面的的词，换成utterance里对应的ptr_x
    """
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
    examples["semantic_parse"] = remove_non_slot_leaf_nodes(examples["semantic_parse"])
    examples["semantic_parse"] = sort_string(examples["semantic_parse"])
    # 修改
    examples["semantic_parse"] = edit_label(examples["semantic_parse"])
    examples["utterance"] = format_time_string(examples["utterance"])
    return examples

def preprocess_dataset(dataset):
    dataset = dataset.map(ptr_change)
    dataset = dataset.filter(ptr_change)
    return dataset

def tokenize_function(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=64, return_tensors="pt")
    tokenized_labels = tokenizer(examples['semantic_parse'], padding='max_length', truncation=True, max_length=64, return_tensors="pt")

    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    return tokenized_inputs

def tokenizer_dataset(tokenizer, dataset):
    tokenized_datasets = dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer})
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


def train(model, dataloader, optimizer, lr_scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播：返回 loss, attention scores 和 logits
        loss, scores, logits = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        # if batch_idx % 10 == 0:
        #     print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} 完成，平均 Loss: {avg_loss:.4f}")


def test_model(model, dataset, device, tokenizer=None):
    # 在 GPU 上测试（如果可用）
    file = open(f"/home/lzx2000/test/low_resources_semantic_parsing/result/decode_{epoch}.txt", "w", encoding="utf-8")
    correct = 0
    data_length = 0
    # DataLoader 用于批量测试
    test_loader = DataLoader(dataset["validation"], batch_size=8, collate_fn=mycollate_trainer)  # 你可以调整 batch_size
    for i, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播：返回 loss, attention scores 和 logits
            loss, scores, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
            # 使用 argmax 获取每个样本的预测类别
            predicted = logits.argmax(-1)
            # 写到文件里
            decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_predicted = tokenizer.batch_decode(predicted, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            for input_id, pred, label in zip(decoded_input, decoded_predicted, decoded_labels):
                file.write(f"input:{input_id}\n")
                file.write(f"pred:{pred}\n")
                file.write(f"label:{label}\n")

            # 检查整个句子是否完全匹配
            correct_predictions = torch.all(torch.eq(predicted, batch["labels"]), dim=1)
            num_correct_sentences = correct_predictions.sum().item()
            correct += num_correct_sentences
            # gz_correct += gz
            # 获取句子的数量
            num_sentences = batch["labels"].size(0)
            data_length = data_length + num_sentences

        # 计算最终准确率
    accuracy = correct / data_length
    return accuracy


def test_model_generate(model, dataset, device, tokenizer=None):
    # 在 GPU 上测试（如果可用）
    # file = open(f"/home/lzx2000/test/low_resources_semantic_parsing/result/decode_{epoch}.txt", "w", encoding="utf-8")
    correct = 0
    data_length = 0
    # DataLoader 用于批量测试
    test_loader = DataLoader(dataset["validation"], batch_size=8, collate_fn=mycollate_trainer)  # 你可以调整 batch_size
    for i, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            print(input_ids.shape)
            # 前向传播：返回 loss, attention scores 和 logits
            predicted= model.generate(input_ids, max_length=64, num_beams=4, num_return_sequences=1)
            # # 使用 argmax 获取每个样本的预测类别
            # predicted = logits.argmax(-1)
            # 写到文件里
            # decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_predicted = tokenizer.batch_decode(predicted, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            for pred, label in zip(decoded_predicted, decoded_labels):
                if pred == label:
                    correct += 1
            # gz_correct += gz
            # 获取句子的数量
            num_sentences = batch["labels"].size(0)
            data_length = data_length + num_sentences

        # 计算最终准确率
    accuracy = correct / data_length
    return accuracy


# 暂时还是不好用
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


# -------------------------------
# 测试示例
# -------------------------------
if __name__ == "__main__":
    model_name = "/data/pretrained_models/roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size


    # 初始化模型（hidden_size 为 1024，feedforward_size 为 512，6层，8头）
    model = RobertaBiLinearAttentionModel(
        roberta_model_name=model_name,
        hidden_size=512,
        num_layers=4,
        num_heads=4,
        vocab_size=vocab_size,
        feedforward_size=128,
        epsilon=0.1,
        dropout=0.1
    )

    import json
    with open('./add_tokens.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    other_tokens = list(data.keys())
    tokenizer.add_tokens(other_tokens)
    model._resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("./data/top_train")
    dataset["train"] = tokenizer_dataset(tokenizer, preprocess_dataset(dataset["train"]))
    # dataset["validation"] = tokenizer_dataset(tokenizer, preprocess_dataset(dataset["validation"]))
    train_loader = DataLoader(dataset["train"], batch_size=4,
                                      collate_fn=mycollate_trainer)  # 你可以调整 batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 1000
    lr = 1e-5
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    # 学习率线性衰减
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    # model_size, factor, warmup, optimizer
    # scheduler = NoamOpt(768, 1.5, 100, optimizer)
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, scheduler, device, epoch)

        # if epoch > 10 and epoch % 100 == 0 :
        #     # accuracy = test_model(model, dataset, device, tokenizer)
        #     accuracy = test_model_generate(model, dataset, device, tokenizer)
        #     print(f"Epoch {epoch} 完成，测试集平均 acc: {accuracy:.4f}")
    #
    # # 保存模型参数
    model.save_pretrained("/home/lzx2000/test/low_resources_semantic_parsing/RoBERTa")
    # print("模型已保存。")


    # 训练，但是完全没有办法去做
    # s = "what is this"
    # # 使用tokenizer将文本转换为token id
    # inputs = tokenizer(s, return_tensors="pt")
    # # 从inputs中获取输入的token id
    # ids = inputs["input_ids"].to(device)
    # print(ids)
    # # 使用模型生成文本
    # outputs = model.generate(ids, max_length=64, num_beams=4, num_return_sequences=4)
    # print(outputs.shape)
    # print(outputs)
    # # 解码生成的token为文本
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #
    # print(generated_text)

    # 最终的测试
    # test_model_generate(model, dataset, device, tokenizer)