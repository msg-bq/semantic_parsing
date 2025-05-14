import sys
from pathlib import Path

# 获取父目录路径并添加到 sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import asyncio

from build_labels import generate_expressions
from generate_natural_language import CustomDataset


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
    return derivation_texts

    # 用断言的方式表达
    # 遵循断言（表达能力更强）的语法结构处理
    # [IN:GET_WEATHER [SL:DATE_TIME Next Monday ] [SL:LOCATION the riverbank ] [SL:WEATHER_ATTRIBUTE hail ] ]
    # al_exps: list[FACT_TYPE] = parse_derivations(derivation_texts, dataset_name)
    #
    # dataset: CustomDataset = generate_nl(al_exps, dataset_name=dataset_name)
    # dataset_in_task_language: CustomDataset = translate_format(dataset, dataset_name=dataset_name)
    #
    # if dataset_name == 'topv2':
    #     dataset: CustomDataset = fix_labels_topv2(dataset_in_task_language)

    return dataset


if __name__ == '__main__':
    dataset_name = 'topv2'
    n = 200
    for i in range(1):
        dataset_test = generate_dataset(dataset_name, n)
        print(dataset_test)
        # with open(f"./other_data_gpt/exchange_top_output_{i}.jsonl", "w", encoding="utf-8") as f:
        #     for e in dataset_test:
        #         import json
        #
        #         json.dump({"input": e.input, "output": e.output}, f)
        #         f.write("\n")
