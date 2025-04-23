import re
import string

import spacy
from remove_non_slot_leaf import remove_non_slot_leaf_nodes

# 加载英文模型，如果处理中文，请使用合适的中文模型
nlp = spacy.load("en_core_web_sm")

true_file = open("event_train6c.tsv", "w", encoding="utf-8")
false_file = open("event_err.tsv", "w", encoding="utf-8")




# 取里面长度大于1的部分
# 名词性短语需要注意一个地方，他有可能被拆开了，如果被拆开了，先直接丢掉
def _extract_noun_phrases(doc):
    """
    利用 spaCy 内置的 noun_chunks 属性提取名词性短语
    """
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    noun_chunks = [chunk for chunk in doc.noun_chunks]

    return noun_phrases, noun_chunks




# event	I'm interested in the third charity event for children featuring Morrissey on November 11, 2022, in Mumbai	[
# IN:GET_EVENT I'm interested in the [SL:ORDINAL third ] [SL:CATEGORY_EVENT charity event ] for [SL:ATTRIBUTE_EVENT
# children ] featuring [SL:NAME_EVENT Morrissey ] on [SL:DATE_TIME November 11, 2022 ] , in [SL:LOCATION Mumbai ] ]
# event	Later this week, is there a free event with Stevie Nicks at the Eiffel Tower?	[IN:GET_EVENT [SL:DATE_TIME
# Later this week ] , is there a [SL:ATTRIBUTE_EVENT free ] event with [SL:NAME_EVENT Stevie Nicks ] at [SL:LOCATION
# the Eiffel Tower ] ? ]

def _get_new_label(origin_label: str, label_replace_words):
    # 将待替换的部分放在第二个捕获组中
    # pattern = r"(\[SL:[A-Za-z0-9_]+\s)(.*?)(\s\])"
    # 这个名字有些看不懂
    pattern = r"(\[SL:[A-Za-z0-9_]+\s)([^[]+?)(\s\])"
    replace_iter = iter(label_replace_words)

    def replacer(match):
        # match.group(1)是"[SL:XXX "，group(2)是待替换部分，group(3)是" ]"
        return match.group(1) + next(replace_iter) + match.group(3)

    return re.sub(pattern, replacer, origin_label)


def _contained_word(word1: str, word2: str) -> bool:
    for word in word1.split():
        if word in word2:
            return True
    return False


# {"input": "What event involves the pet Tylor", "output": "[IN:GET_EVENT What event involves the [SL:ATTRIBUTE_EVENT
# pet ] [SL:NAME_EVENT Tylor ] ]")}


