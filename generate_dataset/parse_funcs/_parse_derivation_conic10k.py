from generate_dataset.modeling import Assertion, BaseOperator, Term, BaseIndividual, Declared_Operators, dummy_operator
from generate_dataset.postprocess.conic10k_ad_hoc import Declaration_record


def __split_operator_variables(derivation_text: str) -> tuple[str, str]:
    """
    Splits the derivation text into the operator name and the variables text.
    Example: "get_sunset([LOCATION:null][DATE_TIME:get_next_day(...)][weather:Rainy])"
    returns ("get_sunset", "[LOCATION:null][DATE_TIME:get_next_day(...)][weather:Rainy]")
    """

    # TODO
    # 对于'InterReciprocal(Eccentricity(CONICSECTION:s)),NumIntersection(LeftPart(HYPERBOLA:d),CURVE:n)'这样的另写,相当于一个算子里的2个输入都是嵌套
    # 目前合成的数据都是直接把这样的丢掉

    start = derivation_text.find("(")
    end = derivation_text.rfind(")")
    if start == -1 or end == -1:
        raise ValueError("Invalid derivation text format")
    operator_name = derivation_text[:start].strip()
    variables_text = derivation_text[start + 1:end].strip()
    return operator_name, variables_text


def __extract_variables_al(variables_text: str, operator: BaseOperator) \
        -> tuple[list[BaseIndividual | Term], list[str]]:
    """
    Extracts variables from the variables text and returns them as a list of BaseIndividual or Term objects.
    Example: "(LOCATION:null, DATE_TIME:get_next_day(...))"
    returns [BaseIndividual("LOCATION", null), Term(get_next_day, [...])]
    """
    variables = []
    declarations = []
    inner_variable_texts = variables_text.split(',')
    merged_inner_variable_texts = __merge_variables(inner_variable_texts)
    for inner_variable_text in merged_inner_variable_texts:
        inner_variable_text = inner_variable_text.strip()
        if "(" in inner_variable_text:  # 说明有嵌套
            sub_operator_name, sub_variables_text = __split_operator_variables(inner_variable_text)
            sub_operator = Declared_Operators[sub_operator_name]
            sub_variables, sub_declarations = __extract_variables_al(sub_variables_text, sub_operator)
            variables.append(Term(operator=sub_operator, variables=sub_variables))
            declarations.extend(sub_declarations)

        else:
            inner_texts = inner_variable_text.split(",")
            for text in inner_texts:
                name = text.split(":")[0]
                value = text.split(":")[-1]

                assert name in operator.inputType + [operator.outputType], \
                    f"Name {name} not found in input types {operator.inputType} or output type {operator.outputType}"

                variables.append(BaseIndividual(value=value))
                declarations.append(f"{value}: {name}")

    return variables, declarations


def __merge_variables(variables_list: list[str]) -> list[str]:
    merged_list = []
    temp_buffer = []
    open_count = 0  # 跟踪未闭合的括号数量

    for item in variables_list:
        # 计算当前项的括号数量
        open_count += item.count('(') - item.count(')')

        # 添加到缓冲区
        temp_buffer.append(item)

        # 如果所有括号都已闭合
        if open_count == 0:
            # 合并缓冲区内容
            merged_item = ''.join([s if i == 0 else f', {s}'
                                   for i, s in enumerate(temp_buffer)])
            merged_list.append(merged_item)
            temp_buffer = []

    # 处理最后可能未闭合的括号
    if temp_buffer:
        merged_list.append(''.join(temp_buffer))

    return merged_list


def _parse_derivation_conic10k(derivation_text: str) -> Assertion | None:
    """
    把生成的表达式转为assertion logic的格式
    Operator(variable) = concept_individual
    """
    derivation_text = derivation_text.replace(' ', '')
    count = derivation_text.count("equals")
    if count >= 2:
        # TODO
        # 这是为了识别 Negation(Term = Term) = BOOL 这样的表达式，可以直接丢掉

        # left_part, sep, right_part = derivation_text.rpartition("equals")
        # left_assertion, left_declarations = _parse_derivation_conic10k(left_part)
        # term_lhs = Term(operator=dummy_operator,
        #                 variables=left_assertion)

        # if "(" in right_part:   # 右式也是一个Operator(variable)的形式
        #     right_operator_name, right_variable_text = __split_operator_variables(right_part)
        #     right_operator = Declared_Operators[right_operator_name]
        #     right_variables, right_declarations = __extract_variables_al(right_variable_text, right_operator)
        #     term_rhs = Term(operator=right_operator,
        #                     variables=right_variables
        #     )
        # else:   # 右式直接是Concept: variable的形式
        #     right_variables = right_part.split(":")[-1]
        #     right_concepts = right_part.split(":")[0]
        #     right_declarations = [f"{right_variables}: {right_concepts}"]
        #     term_rhs = Term(operator=dummy_operator,
        #                     variables=right_variables)
        # return Assertion(lhs=term_lhs, rhs=term_rhs), left_declarations + right_declarations
        return None

    elif count == 0:  # fixme: 为什么会有无equals的情况
        operator_name, variable_text = __split_operator_variables(derivation_text)
        operator = Declared_Operators[operator_name]
        variables, declarations = __extract_variables_al(variable_text, operator)
        term_lhs = Term(operator=operator,
                        variables=variables)
        term_rhs = Term(operator=dummy_operator,
                        variables=[])

        fact = Assertion(lhs=term_lhs, rhs=term_rhs)
        Declaration_record[fact] = declarations
        return fact

    else:
        left_part = derivation_text.split("equals")[0]
        right_part = derivation_text.split("equals")[-1]
        left_operator_name, left_variable_text = __split_operator_variables(left_part)
        left_operator = Declared_Operators[left_operator_name]
        left_variables, left_declarations = __extract_variables_al(left_variable_text, left_operator)
        term_lhs = Term(operator=left_operator,
                        variables=left_variables)

        if "(" in right_part and "EXPRESSION" not in right_part:  # 右式也是一个Operator(variable)的形式
            right_operator_name, right_variable_text = __split_operator_variables(right_part)

            # TODO
            # 需要解决Point(p) = (Number, Number)的特殊右式表达式，目前直接忽略
            if right_operator_name == '':
                return None

            right_operator = Declared_Operators[right_operator_name]
            right_variables, right_declarations = __extract_variables_al(right_variable_text, right_operator)
            term_rhs = Term(operator=right_operator,
                            variables=right_variables
                            )
        else:  # 右式直接是Concept: variable的形式
            right_variables = [BaseIndividual(right_part.split(":")[-1])]
            right_concepts = right_part.split(":")[0]
            right_declarations = [f"{right_variables}: {right_concepts}"]
            term_rhs = Term(operator=dummy_operator,
                            variables=right_variables)

        fact = Assertion(lhs=term_lhs, rhs=term_rhs)
        Declaration_record[fact] = left_declarations + right_declarations
        return fact


if __name__ == '__main__':
    derivation_text_example = \
        ("Coordinate ( LeftFocus ( CONICSECTION: T ) ) equals ( sqrt ( Quadrant ( POINT: t ) ), DotProduct ( VECTOR: "
         "k, VECTOR: D ) )")
    assertions = _parse_derivation_conic10k(derivation_text_example)
    print(f"{'; '.join(Declaration_record[assertions])}\n{assertions}")
