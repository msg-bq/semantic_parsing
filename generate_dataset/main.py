import sys
from pathlib import Path
import json

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
    derivation_texts: list[str] = asyncio.run(generate_expressions(n=num))  # 如果需要靠传参指明路径的话，可以往dataset_file传
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
    # dataset_name = r'D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\rules\topv2_weather.cfg'
    dataset_name = r'D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\rules\conic10k.cfg'
    n = 5000
    for i in range(1):
        dataset_test = generate_dataset(dataset_name, n)
        print(dataset_test)
        with open(f"D:\\桌面\\6023\\generate_dataset\\semantic_parsing-q_upload_change\\generate_dataset\\assertions_output\\conic10k_output_5.jsonl", "a", encoding="utf-8") as f:
            for e in dataset_test:
                json.dump({"assertion": e}, f)
                f.write("\n")
