from .base_classes import Assertion, BaseOperator, Term, BaseIndividual
import generate_dataset.modeling.topv2  # hack: 这里这个实现肯定时不妥当的，需要有个策略导入，比检验
# Declared_Operator必须得先被导入某个建模后，才可以使用
from generate_dataset.modeling.co_namespace import Declared_Operators


def __clean_text(text: str) -> str:
    """
    将一些特殊的转义符号等退回正常的格式。目前先认为仅对variables使用即可（这个是随着parse_derivation自定义的。top的规则里我们不对
    intent使用特殊字符
    """
    text = text.replace('*space', ' ')
    return text


def __split_operator_variables(derivation_text: str) -> tuple[str, str]:
    """
    Splits the derivation text into the operator name and the variables text.
    Example: "get_sunset([LOCATION:null][DATE_TIME:get_next_day(...)][weather:Rainy])"
    returns ("get_sunset", "[LOCATION:null][DATE_TIME:get_next_day(...)][weather:Rainy]")
    """
    start = derivation_text.find("(")
    end = derivation_text.rfind(")")
    if start == -1 or end == -1:
        raise ValueError("Invalid derivation text format")
    operator_name = derivation_text[:start][len('intent:'):].strip()
    variables_text = derivation_text[start + 1:end].strip()
    return operator_name, variables_text


def __extract_variables(variables_text: str, operator: BaseOperator) -> list[BaseIndividual | Term]:
    """
    Extracts variables from the variables text and returns them as a list of BaseIndividual or Term objects.
    Example: "[LOCATION:null][DATE_TIME:get_next_day(...)][weather:Rainy]"
    returns [BaseIndividual("LOCATION", null), Term(get_next_day, [...]), BaseIndividual("WEATHER", "Rainy")]
    """
    def find_matching_bracket(text: str, start: int) -> int:
        """
        Finds the index of the matching closing bracket for a nested structure.
        """
        print("text")
        print(text)

        stack = 1  # The opening bracket at 'start' is already found
        for i in range(start + 1, len(text)):
            if text[i] == '[':
                stack += 1
            elif text[i] == ']':
                stack -= 1
                if stack == 0:
                    return i
        raise ValueError("Mismatched brackets in variables text")

    variables = []
    i = 0
    while i < len(variables_text):
        if variables_text[i] == '[':
            # Find the closing bracket for this variable
            end = find_matching_bracket(variables_text, i)
            inner_text = variables_text[i + 1:end]  # Strip the outer brackets
            name, value = inner_text.split(":", 1)

            assert name in operator.inputType + [operator.outputType], \
                f"Name {name} not found in input types {operator.inputType} or output type {operator.outputType}"

            if value.startswith("intent:"):
                sub_operator_name, sub_variables_text = __split_operator_variables(value)
                sub_operator = Declared_Operators[sub_operator_name]
                sub_variables = __extract_variables(sub_variables_text, sub_operator)
                variables.append(Term(operator=sub_operator, variables=sub_variables))
            else:
                value = __clean_text(value)
                variables.append(BaseIndividual(value=value))

            i = end + 1
        else:
            i += 1

    return variables


def _parse_derivation_topv2(derivation_text: str) -> Assertion:
    """
    把生成的表达式转为topv2的格式
    get_sunset ( [ LOCATION: null ] [ DATE_TIME: get_next_day ( [ datetime: This*spaceMonday ] ) ] [ weather: Rainy ] )
    """
    derivation_text = derivation_text.replace(' ', '')
    operator_name, variables_text = __split_operator_variables(derivation_text)
    operator = Declared_Operators[operator_name]
    variables = __extract_variables(variables_text, operator)
    term_lhs = Term(operator=operator,
                    variables=variables)
    term_rhs = Term(Declared_Operators['dummy_operator'],
                    variables=[])

    return Assertion(lhs=term_lhs, rhs=term_rhs)


if __name__ == '__main__':
    derivation_text_example = \
        'intent:GET_EVENT ( [ CATEGORY_EVENT: null ] [ DATE_TIME: later*spacetoday ] [ LOCATION: null ] [ ATTRIBUTE_EVENT: families ] [ NAME_EVENT: null ] [ ORDINAL: null ] [ ORGANIZER_EVENT: null ] )'
    print(_parse_derivation_topv2(derivation_text_example))
