import copy
from typing import List

def renew_values(values:List[str])->List[str]:
    tmp_values = copy.deepcopy(values)
    for v in values:
        if "(" or ")" in v:         # 说明是嵌套表达式，内部value是一个term
            tmp_values.remove(v)
            op = v.split("(")[0].strip()
            new_value = v.split("(")[1].split(")")[0].strip().split(",")
            tmp_values.append(op)
            tmp_values.append(renew_values(new_value))
        elif v == "None":
            tmp_values.remove(v)
    return tmp_values

def get_values(formal:str)->List[str]:
    formals = formal.split(",")  # 针对复合表达式进行处理
    values_list = []
    for formal in formals:
        formal = formal.strip()
        result = formal.split("=")
        rhs = result[1].strip()
        op = result[0].split("(")[0].strip()
        values = result[0].split("(")[1].split(")")[0].strip().split(",")
        new_values = renew_values(values)
        values_list.append(rhs)
        values_list.append(op)
        values_list += new_values
    return values_list

def check(sentence:str, formal:str) -> bool:
    values = get_values(formal)
    for value in values:
        if value not in sentence or "(" in sentence:
            reture = False
    return True
