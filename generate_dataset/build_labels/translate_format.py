# 本文件用于将断言语法的结果，输出为特定下游任务所需的格式
from parse_funcs import Assertion, Formula, Term, BaseIndividual


def _translate_format_topv2(al_expressions: list[Assertion | Formula]) -> list[str]:
    """
    Converts Assertions or Formulas instance to TOPv2 string format.

    :param al_expressions: Assertion or Formula instances
    :return: Strings representing the instances in TOPv2 format
    """
    def _format(expression: Formula | Assertion | Term | BaseIndividual) -> str:
        if isinstance(expression, Formula):
            return f"{expression.formula_left}; {expression.formula_right}"
        elif isinstance(expression, Assertion):
            return _format(expression.LHS)
        elif isinstance(expression, Term):
            intent = expression.operator.name
            slots = []
            for slot_name, slot_value in zip(expression.operator.inputType, expression.variables):
                formatted_value = _format(slot_value)
                if formatted_value != 'null':
                    slot_str = f"[SL:{slot_name} {formatted_value}]"
                    slots.append(slot_str)
            slots = " ".join(slots)

            return f"[IN:{intent} {slots}]"
        elif isinstance(expression, BaseIndividual):
            return expression.value
        else:
            raise TypeError("Unsupported type for assertion logic. Expected Assertion, Formula, Term or Individual.")

    return [_format(expression) for expression in al_expressions]


translate_format_dict = {'topv2': _translate_format_topv2}


def translate_format(al_expressions: list[Assertion | Formula], dataset_name: str) -> list[str]:
    trans_func = translate_format_dict[dataset_name]
    return trans_func(al_expressions)


if __name__ == '__main__':
    from parse_funcs.parse_derivation import parse_derivations

    derivation_text_example = \
        'intent:get_sunrise ( [ location: intent:get_location ( [ location_user: null ] [ search_radius: null ] [ location_modifier: London ] ) ] [ date_time: Next*spaceFriday ] [ weather: Rainy ] )'
    dataset = 'topv2'
    al_exp = parse_derivations(derivation_text_example, dataset)
    print(translate_format([al_exp], dataset_name=dataset))
