# 考虑输入和label在lemmatization层面的对齐
from generate_dataset.gen_utils.spacy_load import load_spacy_model
from generate_dataset.modeling import FACT_T, Assertion, Formula, Term, BaseIndividual


nlp = load_spacy_model("en_core_web_sm")  # fixme: 重复加载


def _extract_individuals(value: Assertion | Formula | Term | BaseIndividual):
    if isinstance(value, BaseIndividual):
        return value
    elif isinstance(value, Term):
        individuals = []
        for var in value.variables:
            if isinstance(var, BaseIndividual):
                individuals.append(var)
        return individuals
    elif isinstance(value, Assertion):
        return _extract_individuals(value.LHS) + _extract_individuals(value.RHS)
    elif isinstance(value, Formula):
        return _extract_individuals(value.formula_left) + _extract_individuals(value.formula_right) \
            if value.formula_right is not None else _extract_individuals(value.formula_left)
    else:
        raise TypeError


def align_sent_label_by_lemmatization(sentence: str, label: FACT_T) -> FACT_T:
    """
    修复句子和标签不匹配的情况，比如句子中是apples，标签中是apple，可以认为是同一个词，并将标签替换为apples。
    使用词母化 + 根据句子和标签里词母化后的词，把标签里的词改成句子里词母化前的词
    """
    sentence_doc = nlp(sentence)
    # 提取并打印所有词的词母
    sentence_lemmas = [token.lemma_ for token in sentence_doc]
    origin_words = [token.text for token in sentence_doc]

    individuals: list[BaseIndividual] = _extract_individuals(label)  # individual一般对应slots，const/var等相似的含义

    for individual in individuals:
        match = individual.value
        # 先看这个词在sentence里有没有，如果没有再进下面的词母匹配
        if match in origin_words:
            continue

        # 创建一个Doc对象
        match_doc = nlp(match)
        match_lemmas = [token.lemma_ for token in match_doc]
        match_words = [token.text for token in match_doc]

        for j, word in enumerate(match_lemmas):
            if word in sentence_lemmas:
                match_words[j] = sentence_lemmas[sentence_lemmas.index(word)]

        individual.value = " ".join(match_words)

    return label
