import unicodedata
def is_punctuation(char):
    """Check if a character is a punctuation mark."""
    return unicodedata.category(char).startswith('P')

def add_space_after_chinese(text):
    result = ""
    for char in text:
        if '\u4e00' <= char <= '\u9fff' or (char != '.' and is_punctuation(char)):
            result += ' ' + char + ' '
        else:
            result += char 

    result = result.replace("  ", " ").strip()
    return result

def find_long_string_in_list(lst, target_string):
    indexes = []
    for i in range(len(lst)):
        # 将从当前位置开始的列表元素合并成字符串，同时忽略空字符串
        for j in range(i, len(lst)):
            combined = ''.join([x for x in lst[i:j + 1] if x != ''])
            # 如果合并后的字符串与目标字符串匹配
            if combined == target_string:
                # 仅添加非空字符串的索引
                indexes.extend([k for k in range(i, j + 1) if lst[k] != ''])
                return indexes
            # 如果合并后的字符串长度已经超过目标字符串，不再继续
            if len(combined) > len(target_string):
                break
    # if len(indexes) > 2:
    #     indexes = [indexes[0],indexes[-1]]

    return indexes