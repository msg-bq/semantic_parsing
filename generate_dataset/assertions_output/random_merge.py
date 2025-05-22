# 随机组合assertions，得到一个问题

import random
from typing import List, Dict, Tuple, Optional
import json

def preprocess_data(data: List[Dict]) -> List[Dict]:
    """预处理数据，过滤掉declarations或facts为None的元素"""
    return [elem for elem in data if elem.get('declarations') is not None and elem.get('facts') is not None and elem.get('facts').count('(') < 3]

def parse_declarations(decl_str: str) -> Dict[str, str]:
    """解析declarations字符串为字典，过滤数字声明并标准化类型名称"""
    if decl_str is None:
        return {}
    decl_str = decl_str.strip()
    if not decl_str:
        return {}
    decls = {}
    parts = [p.strip() for p in decl_str.split(';') if p.strip()]
    for part in parts:
        try:
            var, typ = part.split(':')
            var = var.strip()
            typ = typ.strip()
            # 过滤掉纯数字的变量名
            if var.isdigit():
                continue
            elif var.startswith('-') and var.lstrip('-').isdigit():
                continue
            elif var.lower() == "true" or var.lower() == "false":
                continue
            # 标准化类型名称：首字母大写，其余小写
            typ = typ.capitalize()
            decls[var] = typ
        except ValueError:
            continue  # 忽略格式错误的声明
    return decls

def check_conflict(decls1: Dict[str, str], decls2: Dict[str, str]) -> bool:
    """检查两个declarations字典是否有冲突"""
    common_vars = set(decls1.keys()) & set(decls2.keys())
    for var in common_vars:
        if decls1[var] != decls2[var]:
            return True
    return False

def combine_elements(elements: List[Dict]) -> Tuple[Dict[str, str], List[str]]:
    """合并多个元素的declarations和facts，返回合并后的declarations和facts列表"""
    combined_decls = {}
    facts = []
    for elem in elements:
        decls = parse_declarations(elem.get('declarations'))
        # 检查冲突
        if check_conflict(combined_decls, decls):
            return None, None
        combined_decls.update(decls)
        fact = elem.get('facts')
        if fact is not None:
            facts.append(fact)
    return combined_decls, facts

def transform_fact_to_question(fact: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    将 fact 的右式替换为 '?'，并返回替换后的 fact 和被替换的右式变量（如果有）。
    例如：
    - 输入 "DotProduct(q, e) = -8" → 返回 ("DotProduct(q, e) = ?", None)
    - 输入 "MajorAxis(k) = o" → 返回 ("MajorAxis(k) = ?", "o")
    """
    if '=' not in fact:
        return None  # 忽略格式错误的 fact
    left, right = fact.split('=', 1)
    right = right.strip()
    
    # 检查右式是否是变量（由字母组成，不含运算符或括号）
    right_var = right if right.isalpha() else None
    transformed_fact = f"{left.strip()} = ?"
    return transformed_fact, right_var

def transform_combination(combination: Dict) -> Dict:
    """
    对单个组合进行转换：
    1. 随机选择一个 fact，将其右式替换为 '?'
    2. 如果被替换的右式是一个变量，则从 declarations 中移除它
    """
    facts = combination["facts"]
    if not facts:
        return combination  # 无 facts 可转换
    
    # 随机选择一个 fact 进行转换
    selected_index = random.randint(0, len(facts) - 1)
    selected_fact = facts[selected_index]
    
    # 替换右式为 '?'
    transformed_fact, right_var = transform_fact_to_question(selected_fact)
    if transformed_fact is None:
        return combination  # 转换失败
    
    # 更新 facts
    new_facts = facts.copy()
    new_facts[selected_index] = transformed_fact
    
    # 更新 declarations（如果右式是变量）
    new_declarations = combination["declarations"].copy()
    if right_var is not None and right_var in new_declarations:
        del new_declarations[right_var]
    
    return {
        "id": combination["id"],
        "declarations": new_declarations,
        "facts": new_facts
    }

def find_valid_combinations(data: List[Dict], min_size=2, max_size=5, max_attempts=5000, max_tuples=10) -> List[Dict]:
    """随机寻找有效的元素组合，确保declarations不超过8个"""
    valid_combinations = []
    attempts = 0
    data = preprocess_data(data)  # 预处理数据
    while attempts < max_attempts and len(valid_combinations) < max_tuples:
        attempts += 1
        k = random.randint(min_size, max_size)
        if k > len(data):
            continue
        selected = random.sample(data, k)
        combined_decls, combined_facts = combine_elements(selected)
        if combined_decls is not None and combined_facts and len(combined_decls) <= 8:
            valid_combinations.append({
                "id": len(valid_combinations) + 1,
                "declarations": combined_decls,
                "facts": combined_facts
            })
    return valid_combinations

def main():
    file = open(r'D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output\assertions\conic10k_output_1.jsonl', 'r', encoding='utf-8')
    data = json.load(file)
    valid_combinations = find_valid_combinations(data, max_tuples=100000)
    
    # 对每个组合进行转换（随机选择一个 fact 替换为 '?'）
    transformed_combinations = [transform_combination(comb) for comb in valid_combinations]
    
    new_file = open(r'D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output\merge_assertions\conic10k_output_1_new.json', 'w', encoding='utf-8')
    json.dump(transformed_combinations, new_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()