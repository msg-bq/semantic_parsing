# 考虑输入和label在lemmatization层面的对齐
from generate_dataset.parse_funcs import Assertion, Formula


def align_sent_label_by_lemmatization(sentence: str, label: Assertion | Formula):
    """
    修复句子和标签不匹配的情况，比如句子中是apples，标签中是apple，可以认为是同一个词，并将标签替换为apples。
    使用词母化 + 根据句子和标签里词母化后的词，把标签里的词改成句子里词母化前的词
    """
    sentence = example["input"]
    label = example["output"]

    sentence_doc = nlp(sentence)
    # 提取并打印所有词的词母
    sentence_lemmas = [token.lemma_ for token in sentence_doc]
    origin_words = [token.text for token in sentence_doc]

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
