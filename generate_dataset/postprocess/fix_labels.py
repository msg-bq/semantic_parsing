from typing import Any

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


def _fill_other_information_topv2(dataset: CustomDataset) -> CustomDataset:
    """
Move the 10am alarm up 30 minutes.
[IN:UPDATE_ALARM Move the [SL:ALARM_NAME [IN:GET_TIME [SL:DATE_TIME 10 am ] ] ] alarm [SL:DATE_TIME up 30 minutes ] . ]
    """
    # todo: 还需要填补非关键信息
    return dataset


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
                return element, st, ed
            else:
                raise ValueError(f"Invalid element type: {type(element)} with the content {element}")

        try:
            reordered_exp, _, _ = _order_expression(parsed_exp, sent)
        except AssertionError as e:
            print(f"AssertionError {e} in _order_expression for sentence: {sent} and parsed_exp: {parsed_exp}")
            example.output = None
            continue

        # Construct the reordered expression back into string form
        def _construct_expression(reordered):
            result = ""
            if isinstance(reordered, tuple):
                element_type, content = reordered
                result += f"[ {element_type} {_construct_expression(content)} ]"
            elif isinstance(reordered, list):  # 对应slots
                values = [f"[ {slot_type} {_construct_expression(slot_value)} ]" for slot_type, slot_value in reordered]
                result += ' '.join(values)
            elif isinstance(reordered, str):
                result += reordered
            return result

        reordered_exp_str = _construct_expression(reordered_exp)

        # Update the label with the reordered expression
        example.output = reordered_exp_str

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
