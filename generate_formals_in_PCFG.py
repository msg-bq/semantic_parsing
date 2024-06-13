import random
import argparse

from generate_individual_func.operators_concepts import operators
from generate_individual_func.from_gpt import call_openai
import generate_individual_func.operators_concepts as operators_concepts

parser = argparse.ArgumentParser(description="Data Generation")
parser.add_argument("--output_file_path",type=str,help="Path of the output.")
parser.add_argument("--generate_by",type=str,default='func',choices=['func','gpt'],help="Choose the way to generate individuals. 'func': generate by self-function, 'gpt': generate by gpt-3.5-turbo.")
parser.add_argument("--n_formals",type=int,default=500,help="Number of formals.")
# parser.add_argument("--n1",type=int,default=0,help="Number of compound expressions.")
# parser.add_argument("--n2",type=int,default=2,help="Number of assertions in one compound expression.")
args = parser.parse_args()


# 创建带算子形式的term
def generate_term_with_operator(term_type="default"):
    '''
    generate term with operator: ops(c1,c2,...)
    '''
    # 嵌套形式
    if term_type != "default":    
        valid_ops = []
        for ops in operators.keys():
            if operators[ops][-1] == term_type:
                valid_ops.append(ops)
        if len(valid_ops) != 0:  # 找到可以进行嵌套的
            operator = random.choice(valid_ops)
        else:                    # 无法进行嵌套,直接返回原子个体
            return generate_atomic_individual(term_type),term_type

    # 初始化时随机选取算子 
    else:                        
        operator = random.choice(list(operators.keys()))

    concepts = []
    print("operator: ", operator)
    for concept in operators[operator][:-1]:
        res = generate_term(concept) 
        concepts.append(res)
        print(concept, ":", res)

    terms = ', '.join(concepts)
    return f"{operator}({terms})", operators[operator][-1]



# 创建原子个体（终止符）
def generate_atomic_individual(concept_name):
    '''
    generate term with atomic individual, i.e. the terminal
    '''
    if args.generate_by == 'gpt':
        return random_from_gpt(concept_name)
    else:
        return random_from_builtin(concept_name)


def random_from_gpt(concept_name)-> str:
    '''
    generate concrete values by gpt.
    '''
    '''
    return a concrete value of the input concept by gpt generating.
    '''
    result = call_openai.generate_from_gpt(concept_name)
    return result


def random_from_builtin(concept_name)-> str:
    '''
    generate concrete values by self-defined function.
    '''
    result = operators_concepts.Concept_name_func_pair[concept_name]()
    return result



def generate_assertion():
    '''
    generate assertions.
    '''
    lhs,rhs_type = generate_term_with_operator()
    rhs = generate_term(rhs_type)
    return f"{lhs} = {rhs}"



def generate_term(term_type):
    '''
    generate terms.

    param: 
    term_type: the type(rhs) of the term

    return:
    nonterminator(nested term) with p=0.3
    terminator(atomic individual) with p=0.7 : 
    normal terminator with p=0.9; 
    special terminator "None" with p=0.1
    '''
    p = random.uniform(0,1)
    if p > 0.7:           # 以 0.3 的概率生成非终止符（嵌套形式）
        res,_ = generate_term_with_operator(term_type)
        return res
    else:                 # 以 0.7 的概率生成终止符
        ps = random.uniform(0,1)
        if ps > 0.9:      # 以 0.1 的概率生成None
            return "None"
        else:            
            return generate_atomic_individual(term_type)



if __name__ == "__main__":
    formals = []

    # 生成实例化断言逻辑表达式
    for _ in range(args.n_formals):               
        formals.append(generate_assertion())
    
    f = open(args.output_file_path,'a',encoding='utf-8')
    for res in formals:
        f.write(res)
        f.write('\n')
    f.close()
