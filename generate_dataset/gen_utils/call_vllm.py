# 将组合的assertions转换成数学问题

import json
import re
from openai import OpenAI

instruction = '''You are a language expert. When I give you a formal text of assertion logic and related assertion logic knowledge, you need to convert it into a mathematical problem.

Assertion logic knowledge：
##### Concept #####
{}

##### Function #####
{}
'''

input_prompt = '''There are some examples.
##### Examples #####
Input assertion logic：
Declarations: "n: Linesegment; m: Hyperbola; y: Conicsection"
Facts: "Abs(n) = -47; Expression(RightPart(m)) = y = 3x + 2"
Query: "UpperFocus(y) = ?"

Output mathematical problem：
"线段n的长为-47，双曲线m右部分的方程为y = 3x + 2，对于圆锥曲线y，其上焦点为什么？"


Input assertion logic：
Declarations: "i: Ellipse; g: Conicsection"
Facts: "Area(g) = 75"
Query: "OverlappingLine(MinorAxis(i)) = ?"

Output mathematical problem：
"圆锥曲线g的面积为75，对于椭圆i，其短轴所在的直线为什么？"


Now please convert the following assertion logic into a mathematical problem.
Input assertion logic：
Declarations: {}
Facts: {}
Query: {}

Output mathematical problem：
'''


def find_concepts(concepts, concepts_name):
    concept_prompt = []
    for index, concept in enumerate(concepts_name):
        if concept in concepts.keys():
            concept_prompt.append(f"{index + 1}. {concept.capitalize()}: {concepts[concept.capitalize()]}")
    return '\n'.join(concept_prompt)


def find_operators(operators, operators_name):
    opeartor_prompt = []
    for index, operator in enumerate(operators_name):
        if operator in operators.keys():
            opeartor_prompt.append(f"{index + 1}. {operator.capitalize()}: {operators[operator.capitalize()]}")
    return '\n'.join(opeartor_prompt)


if __name__ == '__main__':
    client = OpenAI(api_key="xxx", base_url="https://ark.cn-beijing.volces.com/api/v3")
    file = open(
        r'D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output'
        r'\merge_assertions\conic10k_output_1_new.json',
        'r', encoding='utf-8')
    datas = json.load(file)

    operator_file = open(r'../ad_hoc_(can_ignore)/operators.json', 'r', encoding='utf-8')
    all_operators = json.load(operator_file)

    concept_file = open(r'../ad_hoc_(can_ignore)/concepts.json', 'r', encoding='utf-8')
    all_concepts = json.load(concept_file)

    new_file = open(
        r'D:\桌面\6023\generate_dataset\semantic_parsing-q_upload_change\generate_dataset\assertions_output\problems',
        'a', encoding='utf-8')

    res = []
    pattern = r'([A-Za-z_]\w*)\('
    for i, data in enumerate(datas):
        dec = data["declarations"]
        facts = data["facts"]
        final_facts = []
        for fact in facts:
            if "?" in fact:
                query = fact
            else:
                final_facts.append(fact)
        declarations = []
        concepts_name = []
        for key, value in dec.items():
            if value == "Expression":
                continue
            declarations.append(f"{key}: {value}")
            concepts_name.append(value)
        function_names = re.findall(pattern, '; '.join(facts))
        concepts_knowledge = find_concepts(all_concepts, list(set(concepts_name)))
        operators_knowledge = find_operators(all_operators, list(set(function_names)))
        assertion_knowledge = instruction.format(concepts_knowledge, operators_knowledge)
        prompts = input_prompt.format('; '.join(declarations), '; '.join(final_facts), query)
        # print(assertion_knowledge + '\n\n' + prompts)
        # prompt = assertion_knowledge + '\n\n' + prompts

        completion = client.chat.completions.create(
            model="deepseek-v3-241226",
            messages=[
                {"role": "system", "content": assertion_knowledge},
                {"role": "user", "content": prompts},
            ],
        )
        ans = completion.choices[0].message.content
        data["math_problem"] = ans

        print(ans)
        print('########################################################\n')
        res.append(data)

    json.dump(res, new_file, indent=4, ensure_ascii=False)
