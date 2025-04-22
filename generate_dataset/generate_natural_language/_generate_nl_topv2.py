import warnings

from ..gen_utils.access_llm import query_gpt

# Instructions template for natural language generation
generate_nl_instruct = '''You are a learned linguist, now please give some natural phrases to represent the meaning of given logical expression.
First, you need to understand the semantics of the input expression. In parentheses are some of the things you need to include.
Then, you should rewrite it into 3 rich natural sentences with different styles and grammatical rules, where:
The first natural statement can be asked in a direct and concise way, the second sentence can be asked in a different way while maintaining the same semantics and the third sentence can been riched by adding some additional statements appropriately.
Don't give the process, just return the final answer.
Answers should be returned in the following python dictionary format:
{{'expression': "input expressions",
 'sentences': ["first sentence", "second sentence", "third sentence"]  # Each sentence should be a simple sentence with no more than 10 words long.
}}

######### input expression ##########
logical_expression: {logical_expression}
words in expression: {words_in_expression}  # Each sentence of answer dictionary should involve all of these words exactly.
operator description: # Please create words based on the following description.
{operator_description}

######### Answers in dictionary format ##########'''


# Function to extract words, skipping "IN" and "SL" tags
def extract_words_from_expression(expression: str) -> list:
    # Remove the square brackets and split the expression by spaces
    # We'll use regex to match everything that is not an "IN" or "SL" tag or brackets
    expression = expression.replace('[', ' ').replace(']', ' ')

    # Split the cleaned expression into components (by spaces)
    components = expression.split()

    words = []

    for component in components:
        if component.startswith('IN') or component.startswith('SL'):
            continue
        words.append(component)

    return words


def _gen_nl_prompt(expression: tuple[str, str]) -> str:
    words = extract_words_from_expression(expression[0])

    prompt = generate_nl_instruct.format(logical_expression=expression[0],operator_description=expression[1], words_in_expression=' '.join(words))
    return generate_nl_instruct.format(logical_expression=expression[0],operator_description=expression[1], words_in_expression=' '.join(words))


def __clean_and_parse_response(response: str) -> dict:
    response = response.strip()
    response = response.strip('"').strip()
    response = response.rstrip('.').strip()
    response = response.lstrip('logical_expression:').strip()

    try:
        response = response.replace("```python", "").replace("```", "")
        response = response.replace("]}\n}","]}").replace("]}\n]","]}")

        response_dict = eval(response)
        if not isinstance(response_dict, dict):
            raise ValueError("The response is not in the correct format.")

        assert 'expression' in response_dict, "The response does not contain the 'expression' key."
        assert 'sentences' in response_dict, "The response does not contain the 'sentences' key."

        return response_dict
    except Exception:
        print(f"The response is not in the correct format. Response: {response}")


def _valid_response(response_dict: dict, expression: str):
    if response_dict['expression'] != expression:  # todo: 这里感觉提示词就没必要生成一遍expression？
        # 毕竟提示词既没什么额外的修复作用。生成完的expression又不保真
        warnings.warn(f"The expression in the response does not match the input expression. Response expression is"
                      f" '{response_dict['expression']}' while input expression is '{expression}'.")
        response_dict['expression'] = expression   # 目前调成False了

    def __check_original_words(sentence: str, words: list[str]) -> bool:
        # Check if all original words are in the response
        # Check if the word order is preserved
        sentence_words = sentence.lower().split()
        words = [w.lower() for w in words]
        for word in words:
            if word not in sentence_words and word+"?" not in sentence_words and word+"." not in sentence_words and word+"," not in sentence_words:
                warnings.warn(f"Word '{word}' missing in the response sentence {sentence}")
                return False
            # pos = sentence.lower().find(word.lower())
            # if pos == -1:
            #     warnings.warn(f"Word '{word}' missing in the left sentence {sentence}"
            #                   f"while the complete sentence is {complete_sent}.")
            #     return False
            # sentence = sentence[pos + len(word):]

        # 暂时不便校验sentence由几句话组成，我们期望是一句话（凑合的判断条件是，句号+问号+分号+叹号的数量==1）
        return True

    original_words = extract_words_from_expression(expression)
    response_dict['sentences'] = [s for s in response_dict['sentences'] if __check_original_words(s, original_words)]

    return response_dict


def _generate_nl_topv2(label: tuple[str, str]) -> dict:
    prompt = _gen_nl_prompt(label)
    response = query_gpt(prompt)
    result = __clean_and_parse_response(response)

    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        if result:
            result = _valid_response(result, label[0])
            if result['expression'] and result['sentences']:
                return result
        # 若条件不满足，可选择再次查询获取新结果，这里简单假设重新获取response
        response = query_gpt(prompt)
        result = __clean_and_parse_response(response)
        attempt += 1

    return {}  # 若三次尝试都未满足条件，返回空字典
