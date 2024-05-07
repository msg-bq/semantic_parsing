import torch
from collections import defaultdict

from .data_preprocess import PreliminaryDataset


def delete_blank(tokenized_inputs, max_seq=512):
    new_tokenized_inputs = defaultdict(list)
    for example_ids, example_mask in zip(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']):
        new_ids = []
        new_mask = []
        for ids, mask in zip(example_ids, example_mask):
            if ids != 259:
                new_ids.append(ids)
                new_mask.append(mask)
        
        new_ids.extend([0]*(max_seq - len(new_ids)))
        new_mask.extend([0]*(max_seq - len(new_mask)))
        
        new_tokenized_inputs['input_ids'].append(new_ids)
        new_tokenized_inputs['attention_mask'].append(new_mask)

    tokenized_inputs = {k: torch.tensor(v) for k,v in new_tokenized_inputs.items()}
    return tokenized_inputs

from .text_utils import add_space_after_chinese
def tokenize_function(examples, tokenizer):
    # examples = ptr_change(examples)
    examples.natural_sentence =  [add_space_after_chinese(s.replace("得到","") ) for s in examples.natural_sentence]
    tokenized_inputs = tokenizer(examples.natural_sentence, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    tokenized_inputs = delete_blank(tokenized_inputs)
    # print([tokenizer.decode(i) for i in tokenized_inputs['input_ids'][0]])
    tokenized_labels = tokenizer(examples.expression, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

    tokenized_inputs['labels'] = tokenized_labels['input_ids']

    return tokenized_inputs

def tokenize_function_selfTrain(examples, tokenizer):
    # examples = ptr_change(examples)
    examples.natural_sentence =  [add_space_after_chinese(s.replace("得到","") ) for s in examples.natural_sentence]
    tokenized_inputs = tokenizer(examples.natural_sentence, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    tokenized_inputs = delete_blank(tokenized_inputs)
    # print([tokenizer.decode(i) for i in tokenized_inputs['input_ids'][0]])
    tokenized_labels = tokenizer(examples.expression, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

    # tokenized_inputs['labels'] = tokenized_labels['input_ids']

    return tokenized_inputs

tokenizer = None
def tokenizer_dataset(tokenizer, dataset: PreliminaryDataset) -> PreliminaryDataset:
    tokenized_datasets = dataset.map(tokenize_function, tokenizer)
    # tokenized_datasets = tokenized_datasets.remove_columns(["expression"]) # 临时移除
    # tokenized_datasets = tokenized_datasets.remove_columns(["NL"])
    return tokenized_datasets

def tokenizer_dataset_selfTrain(tokenizer, dataset):

    tokenized_datasets = dataset.map(tokenize_function_selfTrain, tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(["expression"])
    tokenized_datasets = tokenized_datasets.remove_columns(["NL"])
    return tokenized_datasets

def mycollate(examples):
    for example in examples:
        for key in example:
            example[key] = torch.tensor(example[key])

    batch = {}
    for key in examples[0]:
        try:
            batch[key] = torch.stack([example[key] for example in examples])
        except:
            batch[key] = [example[key] for example in examples]

    return batch