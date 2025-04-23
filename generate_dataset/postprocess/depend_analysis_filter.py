import spacy
import re

from generate_dataset.generate_natural_language import CustomDataset, Example
from test2 import get_full_noun_label


def _get_new_label(origin_label: str, label_replace_words):
    # 将待替换的部分放在第二个捕获组中
    # pattern = r"(\[SL:[A-Za-z0-9_]+\s)(.*?)(\s\])"
    pattern = r"(\[SL:[A-Za-z0-9_]+\s)([^[]+?)(\s\])"  # fixme: individual instance
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

stopwords = set(open("stopwords.txt", "r", encoding="utf-8").read().splitlines())


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
