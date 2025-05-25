import json
import random

# 读取原始数据
with open(r"D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output\merge_tuples\conic10k_output_1_problem_new.json", "r", encoding="utf-8") as f:
    datas = json.load(f)

res = []
for data in datas:
    declaratiosn = data["Declarations"]
    if "Triangle" in declaratiosn:
        continue
    else:
        res.append(data)

with open(r"D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output\synthetic_datas.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=4, ensure_ascii=False)