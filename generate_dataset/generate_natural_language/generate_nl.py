import random
from pathlib import Path
import json
import sys

from generate_dataset.postprocess.conic10k_ad_hoc import Declaration_record

# 获取父目录路径并添加到 sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)  # 你用这行帮助移除generate_dataset作为模块的一部分
# 不过如果这样的话，感觉不如直接在main.py的位置就导致第一级的目录叭，而不是这里？ fixme(zcl): 你确认一下？我改是可以的
sys.path.append(parent_dir)

from concurrent.futures import ThreadPoolExecutor

from generate_natural_language.dataset_class import CustomDataset, Example
from modeling.base_classes import FACT_TYPE
from postprocess._align_lemma import align_sent_label_by_lemmatization  # fixme: 这样调用有点奇怪
from postprocess._complete_noun_phrases import get_full_noun_label
from generate_natural_language._generate_nl_topv2 import generate_nl_topv2  # fixme: 现在已经没有单独的generate_nl_topv2被使用了，都是通用的方案


generate_nl_func_dict = {'topv2': generate_nl_topv2}


def generate_nl(labels: list[FACT_TYPE], dataset_name: str, workers: int = 200) -> CustomDataset:
    generate_nl_func = generate_nl_func_dict[dataset_name]
    dataset = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_nl_func, label) for label in labels]

        for future in futures:
            result = future.result()
            if result:
                inp = random.choice(result['sentences'])
                out = get_full_noun_label(sentence=inp,
                                          label=align_sent_label_by_lemmatization(sentence=inp,
                                                                                  label=result['expression']))
                example = Example(inp=inp,
                                  # out=result['expression']  # todo, fixme: 这里之后检查一下
                                  out=out)
                dataset.append(example)

    return CustomDataset(dataset)


if __name__ == '__main__':
    from parse_funcs.parse_derivation import parse_derivations
    import json
    file = open(r'D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output\conic10k_output_5.jsonl', 'r', encoding='utf-8')
    new_file = open(r'D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output\conic10k_output_5_new.json', 'a', encoding='utf-8')
    dataset_type = 'conic10k'
    res = []
    for i, line in enumerate(file):
        data = json.loads(line)
        if "}" not in data["assertion"]:
            derivation_text_example = data["assertion"]
            al_exp = parse_derivations(derivation_text_example, dataset_type)
            al_declarations = Declaration_record.get(al_exp, None)
            data["id"] = i
            if al_declarations == None:
                data["declarations"] = None
            else:
                data["declarations"] = '; '.join(al_declarations)
            data["facts"] = str(al_exp)
            json.dump(data, new_file, indent=4, ensure_ascii=False)
            new_file.write(",\n")
            new_file.flush()
            print(i, '   ', al_declarations, al_exp)
    
