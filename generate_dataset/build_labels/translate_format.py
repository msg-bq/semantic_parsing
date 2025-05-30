# 本文件用于将断言语法的结果，输出为特定下游任务所需的格式
from generate_dataset.generate_natural_language.dataset_class import CustomDataset
from generate_dataset.modeling import FACT_TYPE, Assertion, Formula, Term, BaseIndividual


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
                    slot_str = f"[SL:{slot_name} {formatted_value} ] "
                    slots.append(slot_str)
            slots = " ".join(slots)

            return f"[IN:{intent} {slots}]"
        elif isinstance(expression, BaseIndividual):
            return expression.value
        else:
            raise TypeError("Unsupported type for assertion logic. Expected Assertion, Formula, Term or Individual.")

    return [_format(expression) for expression in al_expressions]  # 这里有个小瑕疵，我本意希望
# 最外层的入口还是保留get_des()的命名，但不巧我们有Assertion和Formula两个入口，且Assertion也是被调用的一方，就不好单独命名了，因此统一


translate_format_dict = {'topv2': _translate_format_topv2}


def translate_format(dataset: CustomDataset, dataset_name: str) -> CustomDataset:
    """
    用于将AL表示的表达式转化为数据集需要的字符串格式
    """
    trans_func = translate_format_dict[dataset_name]

    al_expressions: list[FACT_TYPE] = [d.out for d in dataset]
    converted_exps: list[str] = trans_func(al_expressions)
    for d, e in zip(dataset, converted_exps):
        d.out = e

    return dataset


if __name__ == '__main__':
    from ..parse_funcs.parse_derivation import parse_derivations

    derivation_text_example = \
        ('intent:GET_SUNRISE ( [ LOCATION: intent:GET_LOCATION ( [ LOCATION_USER: XXX ] [ SEARCH_RADIUS: null ] [ '
         'LOCATION_MODIFIER: null ] ) ] [ DATE_TIME: null ] )')
    dataset_type = 'topv2'
    al_exp = parse_derivations(derivation_text_example, dataset_type)
    # ['[IN:GET_SUNRISE [SL:LOCATION [intent:GET_LOCATION([LOCATION_USER:Toronto][SEARCH_RADIUS:null][
    # LOCATION_MODIFIER:null])]] [SL:DATE_TIME yesterday]]']]
    print(_translate_format_topv2([al_exp]))
