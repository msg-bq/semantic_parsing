def sort_string(s):
    """
    输入一个类似下面格式的字符串：
      [IN:GET_WEATHER [SL:WEATHER_ATTRIBUTE warm ] [SL:DATE_TIME tomorrow morning ] [SL:WEATHER_TEMPERATURE_UNIT fahrenheit ] ]
    输出排序后的结果，排序规则为：在每个 [IN:…] 块内，将直接子项中所有 [SL:XXX ...] 按照 XXX（即冒号后第一个单词）的字母序排序，
    如果多个 [SL:XXX ...] 的 XXX 部分相同，则保持原有顺序。
    嵌套的 [IN:…] 块独立处理。
    """
    token, _ = parse_token(s, 0)
    sort_token(token)
    return token_to_string(token)


def parse_token(s, i):
    """
    递归解析，从 s[i] 开始解析一个 token（应以 '[' 开头），返回 (token, new_index)。
    token 以字典表示，结构为：
      {
         'type': "IN" 或 "SL",
         'command': 第一个单词（如 "GET_WEATHER" 或 "WEATHER_ATTRIBUTE"），
         'args': 剩余的纯文本参数（可能为空），
         'children': 直接嵌套的子 token 列表
      }
    """
    assert s[i] == '[', f"预期 '[' 开头，当前字符：{s[i]}"
    i += 1  # 跳过 '['

    # 读取 token 类型，直到遇到冒号
    token_type = ""
    while i < len(s) and s[i] != ':':
        token_type += s[i]
        i += 1
    token_type = token_type.strip()
    i += 1  # 跳过冒号

    # 读取 command（第一个单词），直到遇到空白、'[' 或 ']'
    command = ""
    while i < len(s) and s[i] not in [' ', '\t', '\n', '[', ']']:
        command += s[i]
        i += 1
    command = command.strip()

    # 跳过 command 后的空白
    while i < len(s) and s[i].isspace():
        i += 1

    args = ""
    children = []

    # 读取 token 内部内容，直到遇到对应的 ']'
    while i < len(s) and s[i] != ']':
        if s[i] == '[':
            # 如果遇到子 token，则递归解析
            child, i = parse_token(s, i)
            children.append(child)
            # 解析完后跳过可能的空白
            while i < len(s) and s[i].isspace():
                i += 1
        else:
            # 否则读取纯文本（参数部分），直到遇到 '[' 或 ']'
            start = i
            while i < len(s) and s[i] not in ['[', ']']:
                i += 1
            args += s[start:i]
            # 去掉首尾多余空格，但注意保留中间的空格
            # （这里可以根据需要调整，默认用 strip()）
            args = args.strip()  # 可选，根据需要是否保留原始空白
            # 如果后面还有空格则跳过
            while i < len(s) and s[i].isspace():
                i += 1

    # 跳过 ']'
    i += 1

    token = {
        'type': token_type,
        'command': command,
        'args': args,
        'children': children
    }
    return token, i


def sort_token(token):
    """
    对 token 进行排序：
      如果 token 是 [IN:...] 类型，则将其直接 children 中所有类型为 SL 的 token 按其 command 字母序排序（保持稳定性）。
      同时对子 token 递归调用 sort_token。
    """
    # 先对子 token 递归排序
    for child in token['children']:
        sort_token(child)

    if token['type'] == 'IN':
        # 只对直接 children 中类型为 SL 的 token 排序，
        # 若 children 中存在其他类型的 token，则保持它们在原位置不变。
        # 记录 SL token 的原始索引及 token 对象
        sl_indices = []
        sl_tokens = []
        for idx, child in enumerate(token['children']):
            if child['type'] == 'SL':
                sl_indices.append(idx)
                sl_tokens.append(child)
        # 对 sl_tokens 按 command 排序（稳定排序）
        sl_tokens_sorted = sorted(sl_tokens, key=lambda x: x['command'])
        # 将排序后的 SL token 放回原来的索引位置
        for pos, sorted_token_obj in zip(sl_indices, sl_tokens_sorted):
            token['children'][pos] = sorted_token_obj


def token_to_string(token):
    """
    将 token 按照格式重构为字符串。
    输出格式示例：
      [IN:GET_WEATHER [SL:DATE_TIME tomorrow morning ] [SL:WEATHER_ATTRIBUTE warm ] [SL:WEATHER_TEMPERATURE_UNIT fahrenheit ] ]
    """
    parts = [token['command']]
    if token['args']:
        parts.append(token['args'])
    # 如果有 children，则每个子 token前加一个空格分隔
    for child in token['children']:
        parts.append(token_to_string(child))
    inner = " ".join(parts)
    return f"[{token['type']}:{inner} ]"


# ---------------------------
# 以下为简单测试代码
if __name__ == "__main__":
    # 示例1
    # input_str = "[IN:GET_WEATHER [SL:WEATHER_ATTRIBUTE warm ] [SL:DATE_TIME tomorrow morning ] [SL:WEATHER_TEMPERATURE_UNIT fahrenheit ] ]"
    # output_str = sort_string(input_str)
    # print("示例1输出：")
    # print(output_str)
    # # 预期输出：
    # # [IN:GET_WEATHER [SL:DATE_TIME tomorrow morning ] [SL:WEATHER_ATTRIBUTE warm ] [SL:WEATHER_TEMPERATURE_UNIT fahrenheit ] ]
    #
    # # 示例2（包含嵌套）
    # input_str2 = "[IN:GET_WEATHER [SL:LOCATION [IN:GET_LOCATION [SL:LOCATION_USER here ] ] ] [SL:DATE_TIME for next week ] ]"
    # output_str2 = sort_string(input_str2)
    # print("\n示例2输出：")
    # print(output_str2)
    # # 对于嵌套部分：[IN:GET_LOCATION ...] 内部只有一个 token，不受影响；外层仅排序直接的 [SL:DATE_TIME ...] 与 [SL:LOCATION ...]
    file1 = open("preprocess2.txt","w",encoding="utf-8")
    with open("/home/lzx2000/T5-base-lora/TOPv2/low_resource_splits/preprocess.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            file1.write(sort_string(line))
            file1.write("\n")

