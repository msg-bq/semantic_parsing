# 其中一个小实验进行的数据过滤，但是合正常流程和最终的实验无关

import json

# 读取原始数据
with open(r"D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output"
          r"\merge_tuples\conic10k_output_1_problem_new.json", "r", encoding="utf-8") as f:
    datas = json.load(f)

res = []
for data in datas:
    declaration = data["Declarations"]
    if "Triangle" in declaration:  # hack(zcl): 这个是因为cfg里不小心新增了Triangle这个concept，conic10k里没有，
        # 但因为已经生成自然语言问题了，就先这样特殊处理了
        continue
    else:
        res.append(data)

with open(
        r"D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output"
        r"\synthetic_datas.json",
        "w", encoding="utf-8") as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
