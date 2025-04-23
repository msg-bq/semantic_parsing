import re
import string

import spacy
from remove_non_slot_leaf import remove_non_slot_leaf_nodes

# 加载英文模型，如果处理中文，请使用合适的中文模型
nlp = spacy.load("en_core_web_sm")

true_file = open("event_train6c.tsv", "w", encoding="utf-8")
false_file = open("event_err.tsv", "w", encoding="utf-8")


# 英文标点符号
def _format_time_string(input_string):  # fixme(lzx): 这个是不有两遍
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

    def insert_spaces_around_punctuation(s):
        i = 0
        while i < len(s):
            if s[i] in english_punctuation:
                if i > 0 and s[i - 1] != " ":
                    s = s[:i] + " " + s[i:]
                    i += 1  # 跳过新插入的空格
                if (i + 1) < len(s) and s[i + 1] != " ":
                    s = s[:i + 1] + " " + s[i + 1:]
                    i += 1  # 跳过新插入的空格
            i += 1
        return s

    formatted_string = insert_spaces_around_punctuation(formatted_string)
    return re.sub(r"([a-zA-Z0-9])'s", r"\1 's", formatted_string)


# 取里面长度大于1的部分
# 名词性短语需要注意一个地方，他有可能被拆开了，如果被拆开了，先直接丢掉
def _extract_noun_phrases(doc):
    """
    利用 spaCy 内置的 noun_chunks 属性提取名词性短语
    """
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    noun_chunks = [chunk for chunk in doc.noun_chunks]

    return noun_phrases, noun_chunks


# 如果槽中的词是限定词或修饰词，那就不需要了
def _is_insert(chunk, word):
    # 遍历名词性短语中的每个词
    for token in chunk:
        # 判断是否是形容词或限定词（包括数词）
        if token.pos_ == 'ADV' or token.pos_ == 'ADJ' or token.pos_ == 'DET':
            if token.pos_ == 'ADV' or token.pos_ == 'ADJ':
                return True

            if word == token.text or token.text in word:
                return False

    return False

# 取里面长度>1的部分
def _extract_adverbial_phrases(doc):
    """
    提取状语短语：
    遍历文档中所有 token，若 token 的依存标签为 "advmod" 或 "npadvmod"（例如 "this evening" 中 evening 的标签），
    则利用该 token 的 subtree 拼接成短语。
    """
    adv_phrases = []
    for token in doc:
        if token.dep_ in {"advmod", "npadvmod"}:
            # 利用 token.subtree 获取以该 token 为根的所有依存子树
            subtree_tokens = list(token.subtree)
            subtree_tokens.sort(key=lambda t: t.i)
            phrase = " ".join([t.text for t in subtree_tokens])
            adv_phrases.append(phrase)
    return adv_phrases


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


