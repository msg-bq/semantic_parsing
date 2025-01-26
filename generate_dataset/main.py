import asyncio

from build_labels import translate_format, generate_expressions
from generate_natural_language import generate_nl, CustomDataset
from parse_funcs import parse_derivations, Assertion, Formula
from postprocess.fix_labels import fix_labels_topv2


def generate_dataset(dataset_name: str, num: int = 2) -> CustomDataset:
    """
    :param num: 我这边异步比较草率，实际生成的数量可能会略大于num
    """
    derivation_texts: list[str] = asyncio.run(generate_expressions(n=num))
    al_exps: list[Assertion | Formula] = parse_derivations(derivation_texts, dataset_name)
    gen_labels = translate_format(al_exps, dataset_name=dataset_name)
    dataset = generate_nl(gen_labels, dataset_name=dataset_name)

    if dataset_name == 'topv2':
        dataset = fix_labels_topv2(dataset)

    return dataset


if __name__ == '__main__':
    dataset = 'topv2'
    n = 20
    dataset = generate_dataset(dataset, n)
    for e in dataset:
        print(e)
