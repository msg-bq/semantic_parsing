from ._llm_instance_provider import LLMProvider
from ._existed_instance_provider import _get_existed_generators

existed_generators = _get_existed_generators()


concept_list_by_llm = []  # 通过llm实例化的concept列表
concept_list_by_custom_funcs: dict[str: callable] = {}  # 通过自定义的funcs对concept进行实例化


async def get_concept_instance(concept_name: str) -> str | None:
    """
    我们姑且认为不在concept_list_by_custom_funcs里的，都直接走llm，就不维护llm列表了
    """
    if concept_name in concept_list_by_custom_funcs:
        return concept_list_by_custom_funcs[concept_name](concept_name)

    if concept_name in existed_generators:
        return str(existed_generators[concept_name]())  # faker和mimesis的随机数据是含格式的，我们这里默认只要str

    return await LLMProvider.llm_instance(concept_name)


if __name__ == '__main__':
    print(existed_generators.keys())