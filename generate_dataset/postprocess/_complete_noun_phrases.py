import string

from generate_dataset.gen_utils.spacy_load import load_spacy_model
from generate_dataset.modeling import FACT_T, BaseIndividual
from generate_dataset.postprocess._align_lemma import _extract_individuals


nlp = load_spacy_model("en_core_web_sm")
stopwords = set(open(r"C:\Users\YueG_W\Documents\GitHub\semantic_parsing\generate_dataset\postprocess\stopwords.txt", "r", encoding="utf-8").read().splitlines())
# hack: 不知道为什么相对路径不对，先用绝对的了


def _extract_noun_phrases(doc):
    """
    利用 spaCy 内置的 noun_chunks 属性提取名词性短语
    """
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    noun_chunks = [chunk for chunk in doc.noun_chunks]

    return noun_phrases, noun_chunks


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


def _is_same_np(chunk, word):
    """如果两个词的差异仅含限定词或修饰词，认为算同一个词"""
    # 遍历名词性短语中的每个词
    for token in chunk:
        # 判断是否是形容词或限定词（包括数词）
        if token.pos_ == 'ADV' or token.pos_ == 'ADJ' or token.pos_ == 'DET':
            if token.pos_ == 'ADV' or token.pos_ == 'ADJ':
                return True

            if word == token.text or token.text in word:
                return False

    return False


def _contained_word(word1: str, word2: str) -> bool:
    for word in word1.split():
        if word in word2:
            return True
    return False


def _clean_string(text: str) -> str:
    text = text.lower().strip()
    if text[-1] in string.punctuation:
        return text[:-1]
    return text


def get_full_noun_label(sentence: str, label: FACT_T) -> FACT_T:
    """
    结合原句子补全标签中不全面的地方，主要是把标签里的词补成短语。
    例子: todo(lzx): 帮忙补个简短的例子，半截句子也可以


    3.2 通过_extract_noun_phrases和_extract_adverbial_phrases获取名词和状语成分，然后保存下来长度 > 1的短语。
    3.3 找到需要改的地方，通过_get_new_label来得到新的标签。

    ===
    _format_time_string似乎是个特殊情况
    """
    doc = nlp(sentence)

    noun_phrases, noun_chunks = _extract_noun_phrases(doc)
    adv_phrases = _extract_adverbial_phrases(doc)

    noun_phrases: list[str] = [_clean_string(noun_phrase) for noun_phrase in noun_phrases if
                               len(noun_phrase.split()) > 1]
    noun_chunks = [noun_chunk for noun_chunk in noun_chunks if len(noun_chunk.text.split()) > 1]
    adv_phrases = [_clean_string(adv_phrase) for adv_phrase in adv_phrases if len(adv_phrase.split()) > 1]

    individuals: list[BaseIndividual] = _extract_individuals(label)  # individual一般对应slots，const/var等相似的含义
    # fixme: _extract_individuals这个之后不会放到这个函数里

    # 这个matches是个列表
    for i, individual in enumerate(individuals):
        match = individual.value
        # 先在名词性中找对应
        label_match = ""
        if match in adv_phrases:
            # 这里有个问题是Free in Freeway
            continue

        for k, noun_phrase in enumerate(noun_phrases):
            if match in noun_phrase and _is_same_np(noun_chunks[k], match):  # hack: match in noun_phrase针对
                # an apple == apple的情况，但是简单用in无法避免app == an apple的情况
                label_match = noun_phrase
                break

        # if label_match in stopwords:  # fixme(lzx): 可能是label_match - match的部分不能在停用词里
        #     label_replace_words.append(match)

        if label_match:
            individual.value = label_match

        # hack: 需要校验当前individual的修复不会干扰到其他的assertion。比如topv2要求输入和输出包含完全一致的单词，这里
        # 就会校验修正后的名词不会与其他slot的名词重叠。但同理conic10k就不需要
        # 本工作将下面的代码在post_process执行后进行的，所以只需要判断当前slot前后两个slot是否有交集_contained_word
        # todo: 暂定README里提一下吧，另外topv2的要求也可以暂时不实现，毕竟已经生成过了。加个fixme: 这里有个忽略topv2的地方，没写检查
        # 感觉可以弄到最终去做一次检验，单独放一个函数。无非就是有些浪费nl的pair

        # if i != 0 and _contained_word(matches[i - 1], label_match):  # topv2才需要
        #     # 其他的任务是否需要count
        #     label_replace_words.append(match)
        #
        # if i != len(matches) - 1 and _contained_word(matches[i + 1], label_match):
        #     label_replace_words.append(match)

    return label

# Bob's teacher is Alice.
