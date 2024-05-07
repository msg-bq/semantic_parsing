

def split_expression(expression):
    """
    将表达式分割成谓词和参数
    :param expression: 表达式
    :return: 谓词，参数
    """
    expression = expression.replace(" ", "")
    predicate = expression.split("(")[0]
    arguments = expression.split("(")[1][:-1].split(",")
    return predicate, arguments