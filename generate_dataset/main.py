import asyncio

from build_labels import translate_format, generate_expressions
from generate_natural_language import generate_nl, CustomDataset
from parse_funcs import parse_derivations, Assertion, Formula
from postprocess.topv2_process_fix_labels import fix_labels_topv2
import torch
torch.utils.data


# ==============
# 需要想个办法可以便捷地注册对应的ops
# Declared_Operators.update()

# ==============

def generate_dataset(dataset_name: str, num: int = 2) -> CustomDataset:
    """
    :param num: 我这边异步比较草率，实际生成的数量可能会略大于num
    """
    # 生成完后的标签了
    derivation_texts: list[str] = asyncio.run(generate_expressions(n=num))

    # 用断言的方式表达
    # 遵循断言（表达能力更强）的语法结构处理
    # [IN:GET_WEATHER [SL:DATE_TIME Next Monday ] [SL:LOCATION the riverbank ] [SL:WEATHER_ATTRIBUTE hail ] ]
    al_exps: list[Assertion | Formula] = parse_derivations(derivation_texts, dataset_name)

    gen_labels: list[tuple[str,str]] = translate_format(al_exps, dataset_name=dataset_name)
    dataset: CustomDataset = generate_nl(gen_labels, dataset_name=dataset_name)

    if dataset_name == 'topv2':
        dataset: CustomDataset = fix_labels_topv2(dataset)

    return dataset


if __name__ == '__main__':
    dataset_name = 'topv2'
    n = 200
    for i in range(1):
        dataset_test = generate_dataset(dataset_name, n)
        with open(f"./other_data_gpt/exchange_top_output_{i}.jsonl","w",encoding="utf-8") as f:
            for e in dataset_test:
                import json
                json.dump({"input":e.input,"output":e.output},f)
                f.write("\n")

