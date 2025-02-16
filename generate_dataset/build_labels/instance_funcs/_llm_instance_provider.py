# 把基于llm的实例化封在一个单独的provider里面。这样我们实现点保存之类的功能
import os.path
import random
from collections import defaultdict
from pathlib import Path

from mimesis.providers.base import BaseProvider
from ._instance_prompt import _concept_instance_prompt
from utils.access_llm import async_query_gpt

_DATADIR = Path(__file__).parent / 'llm_instance_datadir'
if not os.path.exists(_DATADIR):
    os.makedirs(_DATADIR)


def _clean_instance_response(response: str) -> str:
    response = response.strip()
    response = response.strip('"')
    response = response.rstrip('.')
    return response


def _read_concept_file(path: str) -> list[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _write_concept_file(path: str, concept_instances: list[str]) -> None:
    # 避免写入重复元素
    existed_concept_instances = set(_read_concept_file(path)) if os.path.exists(path) else set()
    concept_instances = set(concept_instances) & existed_concept_instances
    with open(path, 'w') as f:
        for concept_instance in concept_instances:
            f.write(concept_instance + '\n')


def _random_choice_and_delete(candidate: list[str]) -> str:
    if candidate:
        n = random.randint(0, len(candidate)-1)
        value = candidate.pop(n)
        return value

    raise ValueError("candidate is empty")


_CONCEPT_CANDIDATES = defaultdict(list)  # hack: 这里的实现有点丑，我是期望可以边用边删除用过的concept instance，
# 但又不希望靠删除文件的行来实现。所以在每次运行期间专门维护一个


async def _llm_concept_instances(concept_name: str) -> list[str]:
    prompt = _concept_instance_prompt(concept_name)
    while True:  # hack: 这里给个最大的try_cnt会好一点
        try:
            concept_instances_text = await async_query_gpt(prompt, temperature=0.3)
            concept_instances_text = _clean_instance_response(concept_instances_text)
            concept_instances = [c.strip()
                                 for c in concept_instances_text.lstrip('[').rstrip(']').split(',')]
            if concept_instances:
                return concept_instances
        except Exception as e:
            print(e)
            pass


class LLMProvider(BaseProvider):

    class Meta:
        name = "llm_provider"

    @staticmethod
    async def llm_instance(concept_name: str = "weather") -> str:
        concept_path = os.path.join(_DATADIR, f"{concept_name}.txt")

        if os.path.exists(concept_path):
            if concept_name not in _CONCEPT_CANDIDATES:  # 只有每次运行的第一次需要读入，后面的candidate list
                # 是靠_CONCEPT_CANDIDATES维护的，就不需要读入了，只是有很多写入
                _CONCEPT_CANDIDATES[concept_name] = _read_concept_file(concept_path)

            # 当已有值不足5个时，才去利用llm生成
            if len(_CONCEPT_CANDIDATES[concept_name]) < 5:
                new_instances = await _llm_concept_instances(concept_name)
            else:
                return _random_choice_and_delete(_CONCEPT_CANDIDATES[concept_name])

        else:
            new_instances = await _llm_concept_instances(concept_name)

        if new_instances:
            _CONCEPT_CANDIDATES[concept_name].extend(new_instances)
            _write_concept_file(concept_path, new_instances)

        return _random_choice_and_delete(_CONCEPT_CANDIDATES[concept_name])
