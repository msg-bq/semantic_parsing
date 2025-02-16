from typing import Any, List
import sys
import os
import re
import string
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_natural_language.generate_nl import CustomDataset, Example


def _parse_expression_topv2(expression: str) -> tuple:
    # Example: [IN:get_sunrise [SL:location [IN:get_location [SL:location_modifier London]]] \
    # [SL:date_time Next Friday] [SL:weather Rainy]]
    # We assume a very basic structure for parsing here (to be improved with more robust methods)
    """
    输出格式如下：
    ('IN:get_sunrise',
      [('SL:location',
        [('IN:get_location', [('SL:location_modifier', ['London'])])]),
       ('SL:date_time', ['Next Friday']),
       ('SL:weather', ['Rainy'])])
    """
    parsed = []
    stack = []
    i = 0
    while i < len(expression):
        if expression[i] == '[':
            # Start of a new IN/SL element, capture the type
            j = i + 1
            while expression[j] != ' ' and expression[j] != '[':
                j += 1
                if j == len(expression):
                    print("expression是这个", expression)
            element_type = expression[i + 1:j]
            stack.append((element_type, []))
            i = j
        elif expression[i] == ']':
            # End of an IN/SL element, pop and save it
            element_type, content = stack.pop()
            if stack:
                stack[-1][1].append((element_type, content))
            else:
                parsed.append((element_type, content))
            i += 1
        else:
            # Capture the value
            j = i
            while i < len(expression) and expression[i] not in ['[', ']']:
                i += 1
            value = expression[j:i].strip()
            if stack and value:
                stack[-1][1].append(value)
    return parsed[0]


# Construct the reordered expression back into string form
def _construct_expression(reordered):
    result = ""
    if isinstance(reordered, tuple):
        element_type, content = reordered
        result += f"[{element_type} {_construct_expression(content)} ]"
    elif isinstance(reordered, list):  # 对应slots
        # values = [f"[ {slot_type} {_construct_expression(slot_value)} ]" for slot_type, slot_value in reordered]
        values = []
        for slot in reordered:
            if isinstance(slot, tuple):
                slot_type, slot_value = slot
                values.append(f"[{slot_type} {_construct_expression(slot_value)} ]")
            else:
                # 非关键信息
                values.append(slot)
        result += ' '.join(values)
    elif isinstance(reordered, str):
        result += reordered
    return result


# 按照topv2的特殊格式转换input字符串
def _format_time_string(input_string: str) -> str:
    english_punctuation = string.punctuation.replace(':', '')
    # 正则表达式匹配时间格式
    time_pattern = r'(\d{1,2})(:)?(\d{2})?(:)?(\d{2})?(am|pm|AM|PM)?'
    # 替换逻辑
    def replacer(match):
        hour = match.group(1)
        colon = " : " if match.group(2) else ""
        minute = match.group(3) if match.group(3) else ""
        colon_2 = " : " if match.group(4) else ""
        second = match.group(5) if match.group(5) else ""
        period = f" {match.group(6)}" if match.group(6) else ""
        return f"{hour}{colon}{minute}{colon_2}{second}{period}"
    # 对字符串进行替换
    formatted_string = re.sub(time_pattern, replacer, input_string)
    # 判断末尾字符是否是english_punctuation内的标点符号且前面无空格
    if formatted_string[-1] in english_punctuation and formatted_string[-2] != " ":
        formatted_string = formatted_string[:-1] + " " + formatted_string[-1]
    # 使用正则表达式找到字母后面的单引号，并在其前面添加一个空格
    return re.sub(r"([a-zA-Z])'s", r"\1 's", formatted_string)


# 从列表元组中抽出来每个槽值
def _extract_last_values(output_lst: List[Any]) -> List[Any]:
    result = []
    def extract_value(element):
        if isinstance(element, list):
            return [extract_value(item) for item in element]
        elif isinstance(element, tuple):
            return extract_value(element[-1])
        else:
            return element

    for item in output_lst:
        result.append(extract_value(item[-1]))

    return result


def _flatten_list(nested_list: List[Any]) -> List[List[str]]:
    # 铺平多层的槽值列表->1维的列表
    def flatten(nested_list: List[Any]) -> List[str]:
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):  # 如果元素是列表，则递归展平
                flat_list.extend(flatten(item))
            else:  # 否则直接添加元素
                flat_list.append(item)
        return flat_list

    flattened = []
    for item in nested_list:
        if not isinstance(item, str):
            flat_list = flatten(item)
            flattened.append(flat_list)
        else:
            flattened.append([item,])

    return flattened


# 设置随机数种子
random.seed(42)
def _change_input_sencenten(sentence: str) -> str:
    if random.random() < 0.85:
        # 替换
        if sentence[-1] in string.punctuation:
            return sentence[:-1]
    return sentence

def _fill_other_information_topv2(dataset: CustomDataset) -> CustomDataset:
    """
Move the 10am alarm up 30 minutes.
[IN:UPDATE_ALARM Move the [SL:ALARM_NAME [IN:GET_TIME [SL:DATE_TIME 10 am ] ] ] alarm [SL:DATE_TIME up 30 minutes ] . ]
    """
    result = []
    for example in dataset:
        # 替换标点符号
        example.input = _change_input_sencenten(example.input)
        # 拆原句input，把pm am : 以及's、标点，给加上空格
        text = _format_time_string(example.input)
        output_lst = example.output[1]
        # 去掉第一个空格前面的operator，以及最后的 "]"
        # [['Rainy'], ['London'], ['Next Friday']]]
        slot_content_lst = _extract_last_values(output_lst)
        slot_content_lst = _flatten_list(slot_content_lst)
        new_example_output = []
        # 3. 在句子中删掉包括这个词在内的前面的词（新建的对象）删掉，继续匹配，
        last_end = 0
        for i, d in enumerate(slot_content_lst):
            sub_sentence = ' '.join(d)
            match = re.search(sub_sentence, text, re.IGNORECASE)
            if match:
                start, end = match.span()  # 获取匹配的起始和结束位置
            else:
                continue
            # 看这个start是不是0，否则就取从0到start-1的位置
            insert_sen = text[last_end:start]
            # 把非关键信息插进去
            if insert_sen.replace(" ", "") != "":
                new_example_output.append(insert_sen.strip())
            # 把原本的SL给插进去
            new_example_output.append(output_lst[i])
            last_end = end
        # 4.直到句子中没词（结束）或者匹配完，句子中还剩下一些（末尾的符号）加到output的最后
        if last_end != len(text):
            new_example_output.append(text[last_end:].strip())

        filled_exp = (example.output[0], new_example_output)
        example.output = _construct_expression(filled_exp)
        
        result.append(example)

    return CustomDataset(result)


def _reorder_expression_topv2(dataset: CustomDataset) -> CustomDataset:
    for example in dataset:
        sent: str = example.input
        exp: str = example.output

        parsed_exp = _parse_expression_topv2(exp)

        # Function to get the position of words in the sentence
        def __get_word_position(word: str, sentence: str) -> tuple[int, int]:
            sentence = sentence.lower()
            assert word.lower() in sentence, f"Word ‘{word}’ not found in sentence ‘{sentence}‘"
            st = sentence.index(word.lower())  # 当一句话中出现多次word时，这里会出现问题。但是修复这一点需要枚举slot的所有排列，不值得
            # 毕竟topv2长度仅在10上下，slot的词也是含义清晰，而很少有停用词之类的
            return st, st + len(word)

        def _order_expression(element: tuple | str, sentence: str) -> tuple[Any, int, int]:
            """
            :param element: 一个expression对应的结构化表达，如
            ('IN:get_sunrise',
              [('SL:location',
                [('IN:get_location', [('SL:location_modifier', ['London'])])]),
               ('SL:date_time', ['Next Friday']),
               ('SL:weather', ['Rainy'])])
            :return: 返回排好序的元素、当前元素的最外的端点（区间），并要求各元素的区间不能有交集
            """
            if isinstance(element, tuple):  # 代表intent或者含有intent的slot
                element_type, content = element
                elem_st, elem_ed = int(1e9), -1
                if isinstance(content, list):
                    elements_with_pos = []  # 记录每个SL的起点和终点
                    if element_type.startswith('IN'):
                        for slot_content in content:
                            ordered_element, sub_st, sub_ed = _order_expression(slot_content, sentence)
                            elements_with_pos.append((ordered_element, sub_st, sub_ed))
                            elem_st, elem_ed = min(elem_st, sub_st), max(elem_ed, sub_ed)

                        # 检查各区间没有交集
                        # 这一步虽然可以放到generate_nl的时候，以期让生成的nl全部符合这一条。但按理说这个情况非常少见（在topv2中）
                        # 我认为可以考虑直接丢弃对应的example，把fix_labels拆出来的清晰度好处高于这几个样例数量的影响
                        elements_with_pos.sort(key=lambda x: x[1])
                        for i in range(len(elements_with_pos) - 1):
                            assert elements_with_pos[i][2] <= elements_with_pos[i + 1][1], \
                                f"Overlapping slots found: {elements_with_pos[i]} and {elements_with_pos[i + 1]}"

                        ordered = [elem[0] for elem in elements_with_pos]
                        return (element_type, ordered), elem_st, elem_ed

                    elif element_type.startswith('SL'):
                        assert len(content) == 1, (f"SL should have only one content, now have {len(content)} "
                                                   f"contents {content}")
                        ordered_elem, sub_st, sub_ed = _order_expression(content[0], sentence)
                        return (element_type, ordered_elem), sub_st, sub_ed  # content取了[0]，是_parse_expression_topv2
                    # 函数给slot的值补了list。但是实际表达式里没有这个。所以return时候也就不用还原list了

                    else:
                        raise ValueError(f"Invalid element type: {element_type} with the content {content}")
                else:
                    raise ValueError(f"Invalid content type: {type(content)} with the content {content}")

            elif isinstance(element, str):  # 代表单纯的slot
                st, ed = __get_word_position(element, sentence)
                return _format_time_string(sentence[st:ed]), st, ed
            else:
                raise ValueError(f"Invalid element type: {type(element)} with the content {element}")

        try:
            reordered_exp, _, _ = _order_expression(parsed_exp, sent)
        except AssertionError as e:
            print(f"AssertionError {e} in _order_expression for sentence: {sent} and parsed_exp: {parsed_exp}")
            example.output = None
            continue

        # Update the label with the reordered expression
        example.output = reordered_exp

    dataset = CustomDataset(data=[d for d in dataset if d.output])
    return dataset


def fix_labels_topv2(dataset: CustomDataset) -> CustomDataset:
    dataset = _reorder_expression_topv2(dataset)
    dataset = _fill_other_information_topv2(dataset)
    return dataset


if __name__ == '__main__':
    expression_test = ('[IN:get_sunrise [SL:location [IN:get_location [SL:location_modifier London]]] '
                       '[SL:date_time Next Friday] [SL:weather Rainy]]')  # XXX: 这个示例好像略有问题，丢了点空格。
    # 反正对_parse_expression_topv2的结果谨慎一下就好
    sent_test = 'Can you tell me the rainy weather forecast in London next Friday?'
    example_test = Example(inp=sent_test, out=expression_test)
    dataset_test = CustomDataset([example_test])
    print("original_exp", expression_test)
    print(fix_labels_topv2(dataset_test)[0])
