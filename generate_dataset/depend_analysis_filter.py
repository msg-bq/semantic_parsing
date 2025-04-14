import spacy
import re
def get_new_label(origin_label: str, label_replace_words):
    # 将待替换的部分放在第二个捕获组中
    # pattern = r"(\[SL:[A-Za-z0-9_]+\s)(.*?)(\s\])"
    pattern = r"(\[SL:[A-Za-z0-9_]+\s)([^[]+?)(\s\])"
    replace_iter = iter(label_replace_words)
    matches = re.findall(pattern, origin_label)

    def replacer(match):
        # match.group(1)是"[SL:XXX "，group(2)是待替换部分，group(3)是" ]"
        return match.group(1) + next(replace_iter) + match.group(3)

    return re.sub(pattern, replacer, origin_label)

# # 处理一个句子
# sentence = "I'm curious if the Shook Twins are performing at an Art Exhibition for infants this Diwali."
#
# label = "[IN:GET_EVENT [SL:NAME_EVENT the Shook Twins ] [SL:CATEGORY_EVENT Art Exhibition ] [SL:ATTRIBUTE_EVENT for infant ] [SL:DATE_TIME Diwali ] ]"

# 加载英语模型
nlp = spacy.load("en_core_web_sm")

# 第一步
def get_sentence_label(example):
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
            for i,lemma in enumerate(sentence_lemmas):
                if lemma == word:
                    s.append(origin_words[i])
                    flag = False
                    break
            if flag:
                s.append(match_words[j])
        # 这里的s即是新的match
        new_match = " ".join(s)
        replace_match.append(new_match)

    new_label = get_new_label(label, replace_match)
    return new_label

# 第二步
# event,me,detail,information,date,what,place
# special_word = ["weather","sunrise","sunset","temperature","me","reminder","time","number","reminders","location","recipe","date","what","place","event","events","delete","it","one","amount","set","times","ones","detail","details"]
special_word = ["event","events","me","detail","details","information","date","what","place"]
# special_word = ["reminder","time","number","reminders","location","recipe","date","what","place","event","events","delete","it","one","amount","set","times","ones","detail","details"]
obj_err_file = open("reminder_obj_err.txt","w",encoding="utf-8")
def obj_int_label(example):
    sentence = example["input"]
    label = example["output"]
    # 解析句子
    doc = nlp(sentence)
    # 这个主要是找到宾语，看是否在标签中，如果不在就丢了这条数据
    # 遍历 Token 并找到宾语
    for token in doc:
        if token.dep_ in ["dobj", "pobj", "obj", "npadvmod"]:
            # print(f"宾语: {token.text}, 依存关系: {token.dep_}, 依赖于: {token.head.text}")
            if token.text.replace(" ","") not in label.replace(" ","") and token.text.lower() not in special_word:
                obj_err_file.write("-----------------\n")
                obj_err_file.write(sentence + "\n")
                obj_err_file.write(f"err:{label}\n")
                obj_err_file.write(f"err:{token.text}\n")
                obj_err_file.write(f"err:{token.dep_}\n")
                return False
    return True


if __name__ == '__main__':
    # {"input": "Please retrieve details for the Ferrari Festival soonest, happening at 1083 Ironwood Walk on Christmas", "output": "[IN:GET_EVENT Please retrieve details for [SL:CATEGORY_EVENT the Ferrari Festival ] [SL:ORDINAL soonest ] , happening at [SL:LOCATION 1083 Ironwood Walk ] on [SL:DATE_TIME Christmas ] ]"}
    import json
    from tqdm import tqdm
    from remove_non_slot_leaf import remove_non_slot_leaf_nodes

    # example = {"input": "Can you tell me about food at noon hip hop parties", "output": "[IN:GET_EVENT Can you tell me about [SL:ATTRIBUTE_EVENT food ] [SL:DATE_TIME at noon ] [SL:CATEGORY_EVENT hip hop parties ] ]"}
    #
    #
    # example = {"input": "Please show off the fourth reminder for March 5, which is a weekday, from the 9 reminders set", "output": "[IN:GET_REMINDER Please [SL:METHOD_RETRIEVAL_REMINDER show off ] the [SL:ORDINAL fourth ] reminder for [SL:TODO March 5 ] , which is a [SL:REMINDER_DATE_TIME weekday ] , from the [SL:AMOUNT 9 ] reminders set ]"}
    #
    # example = {"input": "Is it raining this morning?", "output": "[IN:GET_WEATHER [SL:DATE_TIME: morning] [SL:WEATHER_ATTRIBUTE: rain]]"}

    # # {'domain': 'reminder', 'utterance': 'Did you remind me about the Annual Science Fair at the High School last summer', 'semantic_parse': '[IN:CREATE_REMINDER [SL:PERSON_REMINDED last summer ] [SL:TODO Annual Science Fair at the High School ] [SL:REMINDER_DATE_TIME last summer ] ]'}
    # example["output"] = remove_non_slot_leaf_nodes(example["output"])
    # label = get_sentence_label(example)
    # new_example = {"input": example["input"], "output": label}
    # print(label)
    # from test2 import get_full_noun_label
    # flag, new_label = get_full_noun_label(new_example)
    # example["output"] = new_label

    # print(obj_int_label(example))
    # print(example)

    dataset = []
    with open(
            "/home/lzx2000/test/testgit/event_es_train.jsonl",
            "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    
    new_dataset1 = []
    for data in tqdm(dataset):
        data["output"] = remove_non_slot_leaf_nodes(data["output"])
        label = get_sentence_label(data)
        new_example = {"input": data["input"], "output": label}
    
        from test2 import get_full_noun_label
        flag, new_label = get_full_noun_label(new_example)
    
        if flag == False:
            continue
    
        new_example["output"] = new_label
    
        if obj_int_label(new_example):
            new_dataset1.append(new_example)
    
    with open("template_event_es_train.tsv","w",encoding="utf-8") as f:
        for data in new_dataset1:
            if "this" in data['input'] and "this" not in data['output']:
                continue
            f.write(f"event\t{data['input']}\t{data['output']}\n")