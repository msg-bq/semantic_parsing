#### "random_cfg_deriver.py"
#### Grace Hadiyanto
#### CS439 FA14
import asyncio
import sys
import random


from .instance_funcs import get_concept_instance
from .derivation_funcs import *  # 因为是别人的包，我就按原始的方式直接import *了

extra_nonterminal_chars = ['_']  # 允许下划线成为终止符字符的一部分，并且建议其他地方不再使用下划线作为特殊记号
# hack: 这里需要提醒一下，下划线在小写字符判断时，似乎应该不会出错，而被认为是合理的字符之一。因此不一定需要额外处理
# 但也要考虑到这是否会给reference的获取带来困扰
_NON_TERMINAL_SYMBOL = '$'  # 要求非终止符以$作为起点，并全部大写


def _parse_first_rule(line):
    rule = line.rstrip()
    variable_name, production = rule.split('=')
    return variable_name, production


def _parse_alt_rule(line):
    rule = line.rstrip()
    return rule[1:]


def _find_references(current_string):
    variable_references = []
    begin_index = 0
    end_index = 0
    is_variable = False
    for i, c in enumerate(current_string):
        # If the character is an upper case and the variable flag is off, we've
        # hit the beginning of a new variable, so turn the variable flag on, and
        # reset the begin and end index.
        if c == _NON_TERMINAL_SYMBOL and not is_variable:
            is_variable = True
            begin_index = i
            end_index = i
        # Here we increment the end index of the current variable reference
        # we are counting.
        elif (c.isupper() or c in extra_nonterminal_chars) and is_variable:
            end_index += 1
        # If the character is not an upper case alphabet character and the
        # variable flag is on, it's the end of our variable reference substring.
        # Save the reference to the list and reset the variable flag, begin, and
        # end index.
        elif is_variable:
            if c == ' ':
                variable_references.append(VariableReference(current_string, begin_index, end_index))
                is_variable = False
                begin_index = i
                end_index = i
            else:  # 不允许非终止符直接续终止符相关的形式，所以直接抛异常
                raise SyntaxError(
                    'non-terminal symbol should be only consist of upper characters or some other'
                    f' characters in extra_nonterminal_chars variables. But given {current_string} '
                    f'at position {i} with context '
                    f'{current_string[max(i - 5, 0): min(i + 5, len(current_string))].replace("*space", " ")}'
                )

    # If we've enumerated through the whole string and the variable flag is on,
    # there's a variable that hasn't been added to the list, so add it.
    if is_variable:
        variable_references.append(VariableReference(current_string, begin_index, end_index))
    return variable_references


def _roulette_choice_rule(rules: list[str]) -> str:
    """
    给定一些可能的derivation的rules，以更高的概率选择非嵌套规则，避免嵌套层级过深
    todo: 怎么才能让使用者快速注意到这个函数，然后决定自己想不想用或者更改为random.choice
    """
    non_terminal_symbols_num = [r.count('$') for r in rules]
    z_list = [1 / (y+0.1) for y in non_terminal_symbols_num]  # +0.1是平滑一些，避免算概率时出现0
    total = sum(z_list)
    probability = [z / total for z in z_list]

    return random.choices(rules, weights=probability, k=1)[0]


async def _derive_string(current_string, grammar):
    # While the string is not fully lower case(i.e. contains rules to be replaced
    # with productions):
    # 1. find variable references in the current string
    # 2. choose a random variable reference from the list of references
    # 3. choose a random production to expand from the corresponding variable's rules
    # 4. print logging message
    # 5. update the current string with the random production
    def _avoid_spurious_non_terminal_symbol(terminal_symbol):
        if terminal_symbol.startswith(_NON_TERMINAL_SYMBOL) and terminal_symbol[1:].isupper():
            return terminal_symbol.lower()  # fixme(lbq): 这里为什么有个未使用的
        return terminal_symbol

    # variable_references = []
    updated_string = None
    while True:
        variable_references = _find_references(current_string)
        if not variable_references:
            break

        random_variable = random.choice(variable_references)  # hack: 这个地方酌情优化，未经优化的random会导致大量重复
        # random_production = random.choice(grammar.variable_dict[random_variable.name].rules)
        random_production = _roulette_choice_rule(grammar.variable_dict[random_variable.name].rules)

        terminal_symbols = random_production.split()
        for i, terminal_symbol in enumerate(terminal_symbols):
            if terminal_symbol.endswith('_generate'):
                concept_name = terminal_symbol[:len(terminal_symbol) - len('_generate')]
                concept_instance: str = await get_concept_instance(concept_name.lower().replace('_', ' '))
                # todo: 每次都调取一遍llm太慢了，也容易重复。其实一次性就能返回10个，然后囤起来就是了。
                concept_instance = concept_instance.replace(' ', '*space')
                terminal_symbols[i] = concept_instance
        random_production = ' '.join(terminal_symbols)

        updated_string = (current_string[:random_variable.start_index] + random_production +
                          current_string[random_variable.end_index + 1:])
        print('In "{}" replacing "{}" with "{}" to obtain "{}"'.format(current_string,
                                                                       random_variable.name,
                                                                       random_production,
                                                                       updated_string))
        current_string = updated_string

    return updated_string


async def generate_expressions(n: int) -> list:
    """
    :param n: 生成的数量
    :return 断言逻辑表示的表达式
    """

    if len(sys.argv) != 2:
        print('Required usage: python3 random_cfg_deriver.py <filename>')
        print('Where filename is the name of a .cfg file.')
        exit()

    # Collect command line argument
    filename = sys.argv[1]

    # Parse file into variable objects for our grammar
    print('Loading grammar from "{}"...'.format(filename))
    input_file = open(filename, 'r', encoding='utf8')
    variable_list = []
    current_variable = None
    total_rules = 0
    for line in input_file:
        if line.isspace():
            continue
        elif line[0] == '#':
            continue
        elif line[0] == '|':
            # Arrived at an alternative rule for a variable
            production = _parse_alt_rule(line)
            current_variable.rules.append(production)
            total_rules += 1
        else:
            # Arrived at a new variable rule
            variable_name, production = _parse_first_rule(line)
            current_variable = Variable(variable_name)
            current_variable.rules.append(production)
            variable_list.append(current_variable)
            total_rules += 1
    input_file.close()
    print('Found {} variables and {} total rules.'.format(len(variable_list), total_rules))

    # Instantiate our grammar with the list of variable objects
    the_grammar = Grammar(variable_list)
    start_string = the_grammar.start_variable.name

    # Seed the random generator and initialize an empty list of variable references
    random.seed()

    # Derive a random string from the grammar until all the variables are used
    final_exps = set()
    while True:
        final_string = await _derive_string(start_string, the_grammar)  # todo: 最好这里过滤下全null
        final_exps.add(final_string)
        print('FINAL STRING:\n{}'.format(final_string))

        if len(final_exps) > n:
            break

    return list(final_exps)


if __name__ == '__main__':
    exps = asyncio.run(generate_expressions(n=2))
    print("============")
    print(exps)
