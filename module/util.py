import torch.nn.functional as F
import torch

class PointAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        # self.linear_layer = nn.Linear(250778, 1)


    def forward(self,hidden1_states,hidden2_states,lm_logits,tags = [[0,0,1,2,-1,3,4,5,-1]]):
        taglist = []
        finalBiasList = []
        sentence_len = hidden1_states.shape[1]
        output_sentence_len = hidden2_states.shape[1]
        for tag in tags:
            vect,bias = merge_weights(tag,sentence_len)
            # print('sentence_len', sentence_len)
            taglist.append(vect)
            biasList = []
            for i in range(output_sentence_len):
                biasList.append(bias)
            finalBiasList.append(biasList)
        merge = torch.tensor(taglist, dtype=torch.float32, requires_grad=True).to(hidden1_states.device) # 8 512[实则300]  512[指明需要的行]
        finalBias = torch.tensor(finalBiasList, dtype=torch.float32, requires_grad=True).to(hidden1_states.device)
        # 8 512[实则300] 728

        # print(hidden1_states.shape) #[8, 512, 768]
        attn_weights = torch.matmul(merge,hidden1_states) # hidden1_states1 8 512【300】 728

        attn_weights1 = torch.matmul(hidden2_states, attn_weights.transpose(1, 2))+finalBias   # hidden2_states 8 512 768  attn_weights 8 512 768  attn_weights1 8 512 512
        score = torch.cat([lm_logits,attn_weights1], dim=2)
        # probabilities = F.softmax(score, dim=-1)
        # lm_logits[:, :, -30:] = attn_weights1
        return score
    

def get_vect(i,j,sentence_len=512):
    vect = []
    for t in range(sentence_len):
        vect.append(0)
    t = i
    count = j-i
    while(t<j):
        vect[t] = 1.0/count
        t+=1
    return vect

# 应该要有两个句子长度
def merge_weights(tags,sentence_len):
    vects = []
    count = 0
    num = 0
    previous_tag = None
    bias = []
    for i,tag in enumerate(tags):

        if tag == -1:
            # 按理说这里
            vect = get_vect(i - count - 1-num, i-num,sentence_len)
            if previous_tag is not None:
                vects.append(vect)
                bias.append(0)
                # flag = False
            count = 0
            previous_tag = None
            continue
        # 如果是和前面相同的数字，则累加起来
        if tag == previous_tag:
            count += 1
        else:
            #如果数字和前面不同，则把前面的数字加起来
            vect = get_vect(i-count-1-num, i-num,sentence_len)
            if previous_tag is not None:
                vects.append(vect)
                bias.append(0)
            count = 0
        previous_tag = tag

    final_len = len(vects)
    while final_len < sentence_len:
        vect = get_vect(0,0,sentence_len)
        vects.append(vect)
        final_len+=1
        # 设置成<-1000时不行，目前在-100~-50之间可以
        bias.append(-100)
    # print(vects)
    return vects,bias

def changeModel(labels):
    first = -1
    for i,label in enumerate(labels):
        if label == first:
            labels[i] = -1
        else:
            first+=1
    return labels

from transformers import AutoTokenizer

import re

def is_valid_time_format(input_str):
    # 定义时间格式的正则表达式
    time_pattern = re.compile(r'^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$')

    # 使用正则表达式匹配输入字符串
    match = re.match(time_pattern, input_str)

    # 如果匹配成功，则是有效的时间格式
    return bool(match)

# 定义特殊字符集合
special_tokens = {'</s>', '<pad>', '<unk>',''}

def isdigit(s):
    if s.replace('.', '', 1).isdigit() or (s.find(".")!=-1 and s.replace('.', '', 1) == ""):
        return True
    else:
        return False
def label_words(original_sentence,lst):
    i = 0
    indices = []
    num = 0
    # print(lst)
    while i < len(lst):
        if isdigit(lst[i]):
            # 当前元素是数字字符串，检查下一个元素
            number_str = lst[i]
            indices.append(num)
            i += 1
            while i < len(lst) and isdigit(lst[i]):
                # 下一个元素也是数字字符串，进行拼接
                number_str += lst[i]
                indices.append(num)
                i += 1
            num+=1
        elif lst[i] in special_tokens:  # 特殊字符检查
            indices.append(-1)
            num+=1
            i += 1
        else:
            indices.append(num)
            num+=1
            i += 1
    # print(indices)
    return indices

def label_tokenized_words_corrected(tokenizer,token_list):
    # tokenizer = AutoTokenizer.from_pretrained("/home/lzx/T5-base/tokenizer1/")
    original_sentences = [tokenizer.decode(t[:-1]) for t in token_list]
    lsts = []
    for tokens in token_list:
        decoded_words = [tokenizer.decode([token_id]) for token_id in tokens]
        lsts.append(decoded_words)
    # print(lsts)
    # 遍历分词后的单词
    llables = []
    for i, tokenized_words in enumerate(lsts):
        original_sentence = original_sentences[i]
        labels = label_words(original_sentence,tokenized_words)
        llables.append(labels)
    return llables
