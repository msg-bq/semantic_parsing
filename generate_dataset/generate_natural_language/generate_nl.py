import random
from concurrent.futures import ThreadPoolExecutor

from .dataset_class import CustomDataset
from generate_dataset.modeling.base_classes import FACT_TYPE
from generate_dataset.postprocess._align_lemma import align_sent_label_by_lemmatization  # fixme: 这样调用有点奇怪
from generate_dataset.postprocess._complete_noun_phrases import get_full_noun_label
from ._generate_nl_topv2 import generate_nl_topv2  # fixme: 现在已经没有单独的generate_nl_topv2被使用了，都是通用的方案


generate_nl_func_dict = {'topv2': generate_nl_topv2}


def generate_nl(labels: list[FACT_TYPE], dataset_name: str, workers: int = 200) -> CustomDataset:
    generate_nl_func = generate_nl_func_dict[dataset_name]
    dataset = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_nl_func, label) for label in labels]

        for future in futures:
            result = future.result()
            if result:
                inp = random.choice(result['sentences'])
                out = get_full_noun_label(sentence=inp,
                                          label=align_sent_label_by_lemmatization(sentence=inp,
                                                                                  label=result['expression']))
                example = Example(inp=inp,
                                  # out=result['expression']  # todo, fixme: 这里之后检查一下
                                  out=out)
                dataset.append(example)

    return CustomDataset(dataset)


if __name__ == '__main__':
    from generate_dataset.parse_funcs.parse_derivation import parse_derivations

    derivation_text_example = \
        ('intent:get_sunrise ( [ location: intent:get_location ( [ location_user: null ] [ search_radius: null ] '
         '[ location_modifier: London ] ) ] [ date_time: Next*spaceFriday ] [ weather: Rainy ] )')
    dataset_type = 'topv2'
    al_exp = parse_derivations(derivation_text_example, dataset_type)
    # gen_labels = translate_format([al_exp], dataset_name=dataset_type)  # todo: 现在类型错了
    # print(generate_nl(gen_labels, dataset_name=dataset_type)[0])
