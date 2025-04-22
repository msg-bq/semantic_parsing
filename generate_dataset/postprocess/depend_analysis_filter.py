import spacy
import re

from generate_dataset.generate_natural_language import CustomDataset, Example
from test2 import get_full_noun_label


def _get_new_label(origin_label: str, label_replace_words):
    # 将待替换的部分放在第二个捕获组中
    # pattern = r"(\[SL:[A-Za-z0-9_]+\s)(.*?)(\s\])"
    pattern = r"(\[SL:[A-Za-z0-9_]+\s)([^[]+?)(\s\])"
    replace_iter = iter(label_replace_words)
    # matches = re.findall(pattern, origin_label)

    def replacer(match):
        # match.group(1)是"[SL:XXX "，group(2)是待替换部分，group(3)是" ]"
        return match.group(1) + next(replace_iter) + match.group(3)

    return re.sub(pattern, replacer, origin_label)


# # 处理一个句子
# sentence = "I'm curious if the Shook Twins are performing at an Art Exhibition for infants this Diwali."
#
# label = "[IN:GET_EVENT [SL:NAME_EVENT the Shook Twins ] [SL:CATEGORY_EVENT Art Exhibition ] [SL:ATTRIBUTE_EVENT for
# infant ] [SL:DATE_TIME Diwali ] ]"

# 加载英语模型
nlp = spacy.load("en_core_web_sm")


# 第一步
def align_sent_label_by_lemmatization(example):
    """
    词母化 + 根据句子和标签里词母化后的词，把标签里的词改成句子里词母化前的词
    """
    sentence = example["input"]
    label = example["output"]
    # 创建一个Doc
    sentence_doc = nlp(sentence)
    # 提取并打印所有词的词母
    sentence_lemmas = [token.lemma_ for token in sentence_doc]
    origin_words = [token.text for token in sentence_doc]
    # 创建一个Doc对象
    # 正则表达式，匹配 [SL:...] 中的 BBB 部分
    # pattern = r"\[SL:[A-Za-z0-9_]+\s(.*?)\s\]"
    pattern = r"\[SL:[A-Za-z0-9_]+\s([^[]+?)\s\]"
    # 使用 re.findall 查找所有匹配的 BBB 部分
    matches = re.findall(pattern, label)

    replace_match = []

    for match in matches:
        # 先看这个词在sentence里有没有，如果没有再进下面的词母匹配
        if match in sentence.split() or match in sentence:
            replace_match.append(match)
            continue
        s = []
        # 创建一个Doc对象
        match_doc = nlp(match)
        match_lemmas = [token.lemma_ for token in match_doc]
        match_words = [token.text for token in match_doc]

        for j, word in enumerate(match_lemmas):
            flag = True
            for i, lemma in enumerate(sentence_lemmas):
                if lemma == word:
                    s.append(origin_words[i])
                    flag = False
                    break
            if flag:
                s.append(match_words[j])
        # 这里的s即是新的match
        new_match = " ".join(s)
        replace_match.append(new_match)

    new_label = _get_new_label(label, replace_match)
    return new_label


# 第二步 event,me,detail,information,date,what,place special_word = ["weather","sunrise","sunset","temperature","me",
# "reminder","time","number","reminders","location","recipe","date","what","place","event","events","delete","it",
# "one","amount","set","times","ones","detail","details"]
# special_word = ["event", "events", "me", "detail", "details", "information", "date", "what", "place"]
# hack: 这里怎么理解，以及这个特殊处理在对外宣传时是否可以解释？
# special_word = ["reminder","time","number","reminders","location","recipe","date","what","place","event","events",
# "delete","it","one","amount","set","times","ones","detail","details"]
# obj_err_file = open("reminder_obj_err.txt", "w", encoding="utf-8")
stopwords = set(open("stopwords.txt", "r", encoding="utf-8").read().splitlines())
# import requests
# stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
# stopwords = set(stopwords_list.decode().splitlines())


def keywords_check(example):
    sentence = example["input"]
    label = example["output"]
    # 解析句子
    doc = nlp(sentence)
    # 这个主要是找到宾语，看是否在标签中，如果不在就丢了这条数据
    # 遍历 Token 并找到宾语
    for token in doc:
        if token.dep_ in ["dobj", "pobj", "obj", "npadvmod"]:
            # print(f"宾语: {token.text}, 依存关系: {token.dep_}, 依赖于: {token.head.text}")
            if token.text.replace(" ", "") not in label.replace(" ", "") and token.text.lower() not in stopwords:
                # obj_err_file.write("-----------------\n")
                # obj_err_file.write(sentence + "\n")
                # obj_err_file.write(f"err:{label}\n")
                # obj_err_file.write(f"err:{token.text}\n")
                # obj_err_file.write(f"err:{token.dep_}\n")
                return False
    return True


if __name__ == '__main__':
    # todo(lzx): 能否理一个几个函数的调用链？我看main里面除了三个函数外还有很多其他操作
    # 这些是不可避免的吗？

    # {"input": "Please retrieve details for the Ferrari Festival soonest, happening at 1083 Ironwood Walk on
    # Christmas", "output": "[IN:GET_EVENT Please retrieve details for [SL:CATEGORY_EVENT the Ferrari Festival ] [
    # SL:ORDINAL soonest ] , happening at [SL:LOCATION 1083 Ironwood Walk ] on [SL:DATE_TIME Christmas ] ]"}
    import json
    from tqdm import tqdm
    from utils.remove_non_slot_leaf import remove_non_slot_leaf_nodes

    # example = {"input": "Can you tell me about food at noon hip hop parties", "output": "[IN:GET_EVENT Can you tell
    # me about [SL:ATTRIBUTE_EVENT food ] [SL:DATE_TIME at noon ] [SL:CATEGORY_EVENT hip hop parties ] ]"}
    #
    #
    # example = {"input": "Please show off the fourth reminder for March 5, which is a weekday, from the 9 reminders
    # set", "output": "[IN:GET_REMINDER Please [SL:METHOD_RETRIEVAL_REMINDER show off ] the [SL:ORDINAL fourth ]
    # reminder for [SL:TODO March 5 ] , which is a [SL:REMINDER_DATE_TIME weekday ] , from the [SL:AMOUNT 9 ]
    #  reminders set ]"}
    #
    # example = {"input": "Is it raining this morning?", "output": "[IN:GET_WEATHER [SL:DATE_TIME: morning] [
    # SL:WEATHER_ATTRIBUTE: rain]]"}

    # # {'domain': 'reminder', 'utterance': 'Did you remind me about the Annual Science Fair at the High School last
    # summer', 'semantic_parse': '[IN:CREATE_REMINDER [SL:PERSON_REMINDED last summer ] [SL:TODO Annual Science Fair
    #  at the High School ] [SL:REMINDER_DATE_TIME last summer ] ]'} example["output"] = remove_non_slot_leaf_nodes(
    #  example["output"]) label = get_sentence_label(example) new_example = {"input": example["input"],
    #  "output": label} print(label) from test2 import get_full_noun_label flag, new_label = get_full_noun_label(
    #  new_example) example["output"] = new_label

    # print(obj_int_label(example))
    # print(example)

    dataset = []
    with open(
            "/home/lzx2000/test/testgit/event_es_train.jsonl",
            "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)

    def post_process_dataset(dataset: CustomDataset) -> CustomDataset:  # todo(lzx): 检查下是这个意思吗
        new_dataset1 = []

        for data in tqdm(dataset):  # fixme: tqdm先留着
            data["output"] = remove_non_slot_leaf_nodes(data["output"])
            label_test = align_sent_label_by_lemmatization(data)

            flag_test, new_label_test = get_full_noun_label(sentence=data["input"],
                                                            label=label_test)

            if not flag_test:
                continue

            # new_example = {"input": data["input"], "output": new_label_test}
            new_example = Example(inp=data["input"],
                                  out=new_label_test)

            if keywords_check(new_example):
                new_dataset1.append(new_example)

        return CustomDataset(new_dataset1)

    # with open("template_event_es_train.tsv", "w", encoding="utf-8") as f:
    #     for data in new_dataset1:
    #         if "this" in data['input'] and "this" not in data['output']:
    #             continue
    #         f.write(f"event\t{data['input']}\t{data['output']}\n")
