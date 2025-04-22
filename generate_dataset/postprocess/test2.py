import re
import string

import spacy
from remove_non_slot_leaf import remove_non_slot_leaf_nodes

# 加载英文模型，如果处理中文，请使用合适的中文模型
nlp = spacy.load("en_core_web_sm")

true_file = open("event_train6c.tsv", "w", encoding="utf-8")
false_file = open("event_err.tsv", "w", encoding="utf-8")


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

def get_full_noun_label(sentence: str, label: str) -> tuple[bool, str]:
    """
    结合原句子补全标签中不全面的地方，主要是把标签里的词补成短语
    3.1 remove_non_slot_leaf_nodes （这个好像重复调用了，可以删）。
    3.2 通过_extract_noun_phrases和_extract_adverbial_phrases获取名词和状语成分，然后保存下来长度 > 1的短语。
    3.3 找到需要改的地方，通过_get_new_label来得到新的标签。
    """
    special_words = ["next", "Next", "coming up", "upcoming", "soonest", "new", "first", "second", "third", "fourth",
                     "fifth", "outdoor", "elderly", "family"
                                                    "free", ""]

    label = remove_non_slot_leaf_nodes(label)
    label_replace_words = []
    doc = nlp(sentence)

    noun_phrases, noun_chunks = _extract_noun_phrases(doc)
    adv_phrases = _extract_adverbial_phrases(doc)

    noun_phrases = [format_time_string(noun_phrase) for noun_phrase in noun_phrases if len(noun_phrase.split()) > 1]
    noun_chunks = [noun_chunk for noun_chunk in noun_chunks if len(noun_chunk.text.split()) > 1]
    adv_phrases = [format_time_string(adv_phrase) for adv_phrase in adv_phrases if len(adv_phrase.split()) > 1]

    print(noun_phrases)
    print(adv_phrases)
    # 正则表达式，匹配 [SL:...] 中的 BBB 部分
    # pattern = r"\[SL:[A-Za-z0-9_]+\s(.*?)\s\]"
    pattern = r"\[SL:[A-Za-z0-9_]+\s([^[]+?)\s\]"
    # 使用 re.findall 查找所有匹配的 BBB 部分
    matches = re.findall(pattern, label)

    matches = [format_time_string(match) for match in matches]

    # 这个matches是个列表
    for i, match in enumerate(matches):
        # 先在名词性中找对应
        label_match = ""
        for adv_phrase in adv_phrases:
            # 这里有个问题是Free in Freeway
            if match.lower() == adv_phrase.lower() or match.lower() == adv_phrase[:-1].lower():
                label_match = adv_phrase

        for k, noun_phrase in enumerate(noun_phrases):
            if match in noun_phrase and _is_insert(noun_chunks[k], match):
                label_match = noun_phrase

        if label_match == "":
            label_replace_words.append(match)
        # 看他的前面后面是不是有和他一样的
        elif label_match in special_words:
            label_replace_words.append(match)
        elif i != 0 and _contained_word(matches[i - 1], label_match):
            label_replace_words.append(match)
        elif i != len(matches) - 1 and _contained_word(matches[i + 1], label_match):
            label_replace_words.append(match)
        else:
            label_replace_words.append(label_match)

    new_label = _get_new_label(label, label_replace_words)
    return True, new_label
