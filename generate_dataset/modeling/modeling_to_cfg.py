from collections import defaultdict
from pprint import pprint

from generate_dataset.modeling import BaseOperator
from generate_dataset.modeling.topv2 import all_operators

operators = list(all_operators.values())

cfg_rules = ['$START=$Assertion']


def _generate_lhs_template(operator: BaseOperator) -> str:
    input_text = ", ".join([f"${v}" for v in operator.inputType])
    return f"{operator.name} ( {input_text} )"


def _generate_plain_cfg(operator: BaseOperator) -> str:
    """把算子平铺开来生成cfg规则"""
    lhs_text = _generate_lhs_template(operator)
    template = lhs_text + " equals ${operator.outputType}"
    # print(template)
    return template


def _generate_nest_cfg(ops: list[BaseOperator]) -> list[str]: # :
    """算子参数可以derive的规则，包括正常instance的generate和nest"""
    nest_rules = defaultdict(set)  # 变量名 → derive rule
    for op in ops:
        for var_name in op.inputType:
            var_symbol = f'${var_name}'.upper().replace(' ', '_')  # 理论上var都是大写的，也不允许空格
            # fixme: 下划线我忘了合不合法
            nest_rules[var_symbol].add(f'{var_name}: {var_name.lower()}_generate')

        out_symbol = f'${op.outputType}'.upper().replace(' ', '_')
        nest_rules[out_symbol].add(_generate_lhs_template(op))

    return [f"{var_symbol}={'\n|'.join(rules)}" for var_symbol, rules in nest_rules.items()]


cfg_rules.append('$Assertion=' + '\n|'.join([_generate_plain_cfg(operator) for operator in operators]))
cfg_rules.extend(_generate_nest_cfg(operators))

print(cfg_rules)
