import Levenshtein
from hanlp_common.document import Document
import hanlp

def exec_constituency_parsing(query: str) -> list[str]:
    con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ELECTRA_SMALL)

    def merge_pos_into_con(doc: Document):
        flat = isinstance(doc['pos'][0], str)
        if flat:
            doc = Document((k, [v]) for k, v in doc.items())
        for tree, tags in zip(doc['con'], doc['pos']):
            offset = 0
            for subtree in tree.subtrees(lambda t: t.height() == 2):
                tag = subtree.label()
                if tag == '_':
                    subtree.set_label(tags[offset])
                offset += 1
        if flat:
            doc = doc.squeeze()
        return doc

    pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
    nlp = hanlp.pipeline() \
        .append(pos, input_key='tok', output_key='pos') \
        .append(con, input_key='tok', output_key='con') \
        .append(merge_pos_into_con, input_key='*')

    # If you need to parse raw text, simply add a tokenizer into this pipeline.
    tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    nlp.insert(0, tok, output_key='tok')
    doc = nlp(query)
    tree = doc['con']

    phrase_type = ['NP', 'VP', 'IP', 'QP', 'NP-OBJ', 'NN']
    # https://hanlp.hankcs.com/docs/annotations/constituency/ctb.html 类型逐渐调整
    extracted_phrases = []
    for subtree in tree.subtrees(lambda t: t.label() in phrase_type):
        phrase = "".join(subtree.leaves())
        extracted_phrases.append(phrase)

    return extracted_phrases

def calc_edit_distance(source: str, target: str) -> int:
    special_distance = 0

    edit_distance = Levenshtein.distance(source, target)

    length_diff = len(source) - len(target)
    if length_diff > 0:
        length_distance = length_diff # * 2，本来要*2，但考虑到现在卡了阈值，这个*2会比较尴尬，先忽略
    else:
        length_distance = -length_diff

    if set(source) & set(target) == 0:
        special_distance += 1e9

    return edit_distance + length_distance + special_distance


def _single_label_correct(candidate_labels: list[str], sliver_label: str) -> str:
    candidate_scores = [(calc_edit_distance(sliver_label, label), label) for label in candidate_labels]
    candidate_scores.sort(key=lambda x: x[0])
    print("=====\n", candidate_scores)

    threshold = min(4, max(len(sliver_label), 1))  # 不允许把当前词直接删掉然后换新的词，所以编辑距离不得超过len(sliver_label)

    if candidate_scores[0][0] < threshold:
        return candidate_scores[0][1]
    else:
        return sliver_label

def label_correct(sentence: str, sliver_label: str | list) -> str | list[str]:
    """
    将自训练生成的部分错误sliver label调整为正确的。主要包括：
    1. 错位的，即ptr14-19→ptr15→20
    2. 缺失的，即ptr14-19→ptr15→19
    3. 多余的
    ...

    所以算法逻辑是：
    1. 用最小的编辑距离把A转成一个NP/VP/IP/QP
    2. 然后修改前后长度如果变化，应该增加惩罚项，且长度减少的惩罚＞长度增加
    """
    candidate_labels = exec_constituency_parsing(sentence)
    if isinstance(sliver_label, str):
        return _single_label_correct(candidate_labels, sliver_label)
    elif isinstance(sliver_label, list):
        return [_single_label_correct(candidate_labels, label) for label in sliver_label]
    else:
        raise TypeError

if __name__ == '__main__':
    sentence = "HanLP是面向生产环境的自然语言处理工具包。"
    sliver_labels = ['生产环', '生产环境', '自然语言处', '的自然语言处', '理工具', '自然语言处理工', '环境的', '是面']

    print(label_correct(sentence, sliver_labels))