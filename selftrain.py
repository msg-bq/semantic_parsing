import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import json
import torch
from torch.utils.data import DataLoader

from utils.data_preprocess import read_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

import re
import string
import jsonlines


# 英文标点符号
def format_time_string(input_string):
    english_punctuation = string.punctuation.replace(':', '').replace("'", '')
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

from utils.data_preprocess import edit_label, filter
from utils.remove_non_slot_leaf import remove_non_slot_leaf_nodes
from utils.sort_label import sort_string
def ptr_change1(examples):

    examples["semantic_parse"] = edit_label(examples)["semantic_parse"]
    
    examples["utterance"] = format_time_string(examples["utterance"])
    # 删标签
    examples["semantic_parse"] = remove_non_slot_leaf_nodes(examples["semantic_parse"])
    # 排序
    examples["semantic_parse"] = sort_string(examples["semantic_parse"])
    # print(examples)
    return examples

def preprocess_dataset1(dataset):
    dataset = dataset.filter(filter)
    dataset = dataset.map(ptr_change1, load_from_cache_file=False)

    return dataset

def tokenize_function(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    if examples['semantic_parse'] == None:
        examples['semantic_parse'] = "none"

    tokenized_labels = tokenizer(examples['semantic_parse'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")

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

# "question": input_text, "pred": pred_text,
def unlabel_filter(data,domain):
    weather = ["[IN:UNSUPPORTED_WEATHER", "[IN:GET_WEATHER","[IN:GET_SUNSET","[IN:GET_SUNRISE","[IN:GET_LOCATION","[SL:DATE_TIME","[SL:WEATHER_TEMPERATURE_UNIT","[SL:LOCATION","[SL:WEATHER_ATTRIBUTE","[SL:LOCATION_USER","[SL:SEARCH_RADIUS","[SL:LOCATION_MODIFIER"]
    reminder = ["IN:GET_MESSAGE]","[SL:CONTACT","[SL:MUTUAL_EMPLOYER","[SL:METHOD_RETRIEVAL_REMINDER","[SL:TODO","[SL:RECIPIENT","[IN:DELETE_REMINDER","[SL:CATEGORY_EVENT","[IN:GET_REMINDER","[IN:GET_TODO","SL:ATTENDEE_EVENT]","[IN:GET_RECURRING_DATE_TIME","SL:CONTENT_EXACT]","IN:GET_RECURRING_DATE_TIME]","[SL:PERSON_REMINDED","[IN:SEND_MESSAGE","IN:REPLY_MESSAGE]","[IN:GET_CONTACT","[SL:AMOUNT","SL:CONTACT_RELATED]","SL:FREQUENCY]","SL:DATE_TIME]","[SL:RECURRING_DATE_TIME","[SL:CONTENT_EXACT","[IN:GET_MESSAGE","[SL:DATE_TIME","SL:ATTENDEE]","IN:GET_CONTACT]","SL:RECURRING_DATE_TIME]","[SL:FREQUENCY","[IN:CREATE_REMINDER","[SL:CONTACT_RELATED","[IN:REPLY_MESSAGE","[SL:ATTENDEE_EVENT","[SL:ORDINAL","[SL:ATTENDEE","SL:PERSON_REMINDED]","SL:AMOUNT]","SL:CONTACT]","SL:TYPE_RELATION]","IN:GET_EVENT]","[SL:TYPE_RELATION","[IN:GET_EVENT"]
    event =  ["[SL:CATEGORY_LOCATION","SL:CATEGORY_LOCATION]","[SL:ORGANIZER_EVENT","SL:ORGANIZER_EVENT]","[SL:CATEGORY_EVENT","SL:CATEGORY_EVENT]","[IN:GET_LOCATION","IN:GET_LOCATION]","[SL:ORDINAL","SL:ORDINAL]","[SL:NAME_EVENT","SL:NAME_EVENT]","[SL:LOCATION","SL:LOCATION]","[SL:LOCATION_MODIFIER","SL:LOCATION_MODIFIER]","[SL:DATE_TIME","SL:DATE_TIME]","[SL:ATTRIBUTE_EVENT","SL:ATTRIBUTE_EVENT]","[SL:POINT_ON_MAP","SL:POINT_ON_MAP]","[IN:GET_EVENT","IN:GET_EVENT]",]
    domain_label = {"weather":weather,"event":event,"reminder":reminder}

    label = data['pred']
    for word in label.split():
        if word != ']' and word not in data['question'] and word not in domain_label[domain]:
            return False
    
    return True

from tqdm import tqdm
import torch.nn.functional as F
file_any = open("分析selftrain的作用在our.txt","w",encoding="utf-8")
def get_unlabel_data(save_path: str, domain:str,p_rate: float, analys = True):
    if analys:
        file_any.write("\n\n------------------------------")
    
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
            transition_scores = model.compute_transition_scores(
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
                if analys:
                    label_text = tokenizer.decode(batch["labels"][b_idx], skip_special_tokens=True)
                pred_text = tokenizer.decode(outputs.sequences[b_idx], skip_special_tokens=True)
                conf = confidence_scores[b_idx].item()

                data_dict = {
                    "question": input_text,
                    "pred": pred_text,
                    "confidence": conf
                }
                if unlabel_filter(data_dict,domain):
                    if analys:
                        if pred_text == label_text:
                            true += 1
                        sum += 1
                        data_dict['label'] = label_text
                    all_data.append(data_dict)
        file_any.write(f"{true}/{sum}\n")
        file_any.flush()
    
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


def get_selftrain_model(unlabel_path: str):
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

import random
def get_modeltrain_model(unlabel_train_dataset, model_save_path, lr):

    for data in label_dataset["train"]:
        unlabel_train_dataset.append(data)
    random.shuffle(unlabel_train_dataset)
    
    train_args = TrainingArguments(
        output_dir=f"{model_save_path}/normal",
        num_train_epochs=25,
        per_device_train_batch_size=128,
        learning_rate=lr,
        do_eval=False,
        no_cuda=False
    )

    # 定义Trainer实例
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=mycollate_trainer,
        train_dataset=unlabel_train_dataset
    )

    # 开始训练
    trainer.train()

model_path = "/tmp/ray/session_2025-03-24_19-08-36_660044_1504071/artifacts/2025-03-24_19-08-39/tune_experiment/working_dirs/train_tune_573b5_00000_0_batch_size=128,epoch=1000,learn_rate=0.0000,max_length=128_2025-03-24_19-08-40/mt5-base/checkpoint-63000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 无标注数据
unlabel_dataset = load_dataset("/home/cyz/data/semantic_parsing/code/data/self_train/weather_top")
unlabel_dataset["train"] = tokenizer_dataset(tokenizer, preprocess_dataset1(unlabel_dataset["train"]))
unlabel_train_loader = DataLoader(unlabel_dataset["train"], batch_size=128, collate_fn=mycollate_trainer)  # 你可以调整 batch_size

# 自建的有标注数据
label_dataset = load_dataset("/home/cyz/data/semantic_parsing/code/TOPv2/low_resource_splits/data_1000_new_our")
label_dataset["train"] = tokenizer_dataset(tokenizer, preprocess_dataset1(label_dataset["train"]))
label_dataset["validation"] = tokenizer_dataset(tokenizer, preprocess_dataset1(label_dataset["validation"]))

from testModel import test_model
# 训练模型
# lr = [1e-5, 5e-6, 5e-7, 5e-7, 5e-7,5e-7,5e-7,5e-7,5e-7,5e-7,5e-7,5e-7]

file1 = open("result_our_weather_our.txt","w",encoding="utf-8")
# acc,f1_score = test_model(model, tokenizer, label_dataset, device = device)
# print(f"初始的准确率为{acc}")
# file1.write(f"初始的准确率为{acc}\n")
# file1.write(f"初始的f1_score为{f1_score}\n")

p_rate = 0.2
for i in range(10):
    unlabel_path = "./output_our_weather_our.json"
    model_save_path = "/tmp/ray/selftrain/self_train_weather"
    # 得到数据
    p_rate = get_unlabel_data(unlabel_path,"weather",p_rate)
    # 自训练
    unlabel_train_dataset = get_selftrain_model(unlabel_path)
    # 正常训练
    get_modeltrain_model(unlabel_train_dataset, model_save_path, 1e-5)
    # 看一下在验证集上的准确率
    acc,f1_score = test_model(model, tokenizer, label_dataset, device = device)
    print(f"第{i}个epoch上的准确率为{acc}")
    file1.write(f"第{i}个epoch上的准确率为{acc}\n")
    file1.write(f"第{i}个epoch上的f1_score为{f1_score}\n")
    file1.flush()
    
    # model.save_pretrained(f"{model_save_path}/normal_final_model")
    # model, tokenizer, dataset, args