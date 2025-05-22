import json
import random

# 读取原始数据
with open(r"D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output\synthetic_datas.json", "r", encoding="utf-8") as f:
    data = json.load(f)

assert len(data) == 10456, f"数据数量为 {len(data)}，不是 16500"

# 分层参数设置
num_layers = 8              # 每层 500 条
layer_size = 1307
layers = [data[i * layer_size:(i + 1) * layer_size] for i in range(num_layers)]

# 需要的总样本数
sample_sizes = [2614, 5228, 7842, 10456]

# 每层采样比例 = 样本数 / 总数据数
def stratified_sampling(layers, total_samples, seed=42):
    random.seed(seed)
    samples = []
    per_layer = total_samples // len(layers)  # 均匀抽样数量
    for layer in layers:
        samples.extend(random.sample(layer, per_layer))
    return samples

# 按不同目标数量生成样本
for size in sample_sizes:
    sampled = stratified_sampling(layers, size)
    with open(f"D:\\桌面\\6023\\generate_dataset\\semantic_parsing-q_upload_change\\generate_dataset\\assertions_output\\sample_{size}.json", "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
