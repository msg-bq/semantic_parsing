from utils.access_llm import async_query_gpt
from build_labels.instance_funcs._instance_prompt import _concept_instance_prompt

concept_list_by_llm = []  # 通过llm实例化的concept列表
concept_list_by_custom_funcs: dict[str: callable] = {}  # 通过自定义的funcs对concept进行实例化


def _clean_instance_response(response: str) -> str:
    response = response.strip()
    response = response.strip('"')
    response = response.rstrip('.')
    return response


async def get_concept_instance(concept_name: str) -> str | None:
    """
    我们姑且认为不在concept_list_by_custom_funcs里的，都直接走llm，就不维护llm列表了
    """
    if concept_name in concept_list_by_custom_funcs:
        return concept_list_by_custom_funcs[concept_name](concept_name)

    prompt = _concept_instance_prompt(concept_name)
    while True:  # hack: 这里给个最大的try_cnt会好一点
        try:
            concept_instance = await async_query_gpt(prompt, temperature=0.3)
            concept_instance = _clean_instance_response(concept_instance)
            if concept_instance:
                return concept_instance
        except Exception as e:
            print(e)
            pass
