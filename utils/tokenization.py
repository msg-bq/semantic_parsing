import torch
from collections import defaultdict

from .ExtraNameSpace import DatasetsReaderNameSpace, DatasetsProcessorNameSpace
from .data_preprocess import PreliminaryDataset, ptr_change
from .text_utils import add_space_after_chinese
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer1 = AutoTokenizer.from_pretrained("/data/lbq/models/mt5-base-trained-final-500+500-2-7_again")#("/home/lzx/T5-base/model3/mt5-base-trained-final-500+500-2-7_again")#("/data/lbq/models/mt5-base-trained-final-500+500-2-7_again")#("/data/lbq/models/mt5-base-trained-final-500+500-2-7_again")#

def delete_blank(tokenized_inputs, max_seq=512):
    new_tokenized_inputs = defaultdict(list)
    for example_ids, example_mask in zip(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']):

        for ids, mask in zip(example_ids, example_mask):
            if ids != 259:
                new_tokenized_inputs['input_ids'].append(ids)
                new_tokenized_inputs['attention_mask'].append(mask)

    new_tokenized_inputs['input_ids'].extend([0]*(max_seq - len(new_tokenized_inputs['input_ids'])))
    new_tokenized_inputs['attention_mask'].extend([0]*(max_seq - len(new_tokenized_inputs['attention_mask'])))

    tokenized_inputs = {k: torch.tensor(v) for k,v in new_tokenized_inputs.items()}
    return tokenized_inputs

@DatasetsProcessorNameSpace.register("ours")
def tokenize_function(examples, tokenizer):
    # examples = ptr_change(examples)
    examples.natural_sentence =  [add_space_after_chinese(s.replace("得到","") ) for s in examples.natural_sentence]
    tokenized_inputs = tokenizer(" ".join(examples.natural_sentence), padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    # tokenized_inputs = delete_blank(tokenized_inputs)
    # print([tokenizer.decode(i) for i in tokenized_inputs['input_ids'][0]])
    tokenized_labels = tokenizer(examples.expression, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    tokenized_inputs['expression'] = examples.expression
    tokenized_inputs['natural_sentence'] = examples.natural_sentence

    return tokenized_inputs

@DatasetsProcessorNameSpace.register("topv2")
def tokenize_function(examples, tokenizer):
    examples = ptr_change(examples)
    tokenized_inputs = tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=30, return_tensors="pt")
    tokenized_labels = tokenizer(examples['semantic_parse'], padding='max_length', truncation=True, max_length=30, return_tensors="pt")

    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    return tokenized_inputs

#zcl
def tokenize_function_zcl(examples, tokenizer):
    examples = ptr_change(examples)
    global tokenizer1
    tokenizer = tokenizer1
    tokenized_inputs = tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    tokenized_labels = tokenizer(examples['seqlogical'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")

    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    return tokenized_inputs

@DatasetsProcessorNameSpace.register("zcl")
def tokenize_function(examples, tokenizer):
    return tokenize_function_zcl(examples, tokenizer)

@DatasetsProcessorNameSpace.register("zcl_mixed")
def tokenize_function(examples, tokenizer):
    return tokenize_function_zcl(examples, tokenizer)


@DatasetsProcessorNameSpace.register("ours")
def tokenizer_dataset(tokenizer, dataset: PreliminaryDataset) -> PreliminaryDataset:
    tokenized_datasets = dataset.map(tokenize_function, tokenizer)
    # tokenized_datasets = tokenized_datasets.remove_columns(["expression"]) # 临时移除
    # tokenized_datasets = tokenized_datasets.remove_columns(["NL"])
    return tokenized_datasets

@DatasetsProcessorNameSpace.register("topv2")
def tokenizer_dataset(tokenizer, dataset):
    tokenized_datasets = dataset.map(tokenize_function, tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(["utterance"])
    tokenized_datasets = tokenized_datasets.remove_columns(["semantic_parse"])
    tokenized_datasets = tokenized_datasets.remove_columns(["domain"])

    return tokenized_datasets

@DatasetsProcessorNameSpace.register("zcl")
def tokenizer_dataset(tokenizer, dataset):
    return dataset.map(tokenize_function, tokenizer)

@DatasetsProcessorNameSpace.register("zcl_mixed")
def tokenizer_dataset(tokenizer, dataset):
    return dataset.map(tokenize_function, tokenizer)