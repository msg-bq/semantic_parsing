import random
from concurrent.futures import ThreadPoolExecutor

from ._generate_nl_topv2 import generate_nl_topv2
from torch.utils.data import Dataset


class Example:
    def __init__(self, inp, out, cand_out = None):
        assert out or cand_out

        self.input = inp
        self.output = out
        self._candidate_output: list = cand_out  # fixme: cand_out该不该留

    def __getitem__(self, item):
        if item == 'input':
            return self.input
        elif item == 'output':
            return self.output
        else:
            raise KeyError(f"Invalid key: {item}")

    def __str__(self):
        return f"Input: {self.input}\nOutput: {self.output}"


class CustomDataset(Dataset):
    def __init__(self, data: list[Example]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


generate_nl_func_dict = {'topv2': generate_nl_topv2}


def generate_nl(labels: list[tuple[str, str]], dataset_name: str, workers: int = 200):
    generate_nl_func = generate_nl_func_dict[dataset_name]
    dataset = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_nl_func, label) for label in labels]

        for future in futures:
            result = future.result()
            if result:
                example = Example(inp=random.choice(result['sentences']),
                                  out=result['expression'])
                dataset.append(example)

    return CustomDataset(dataset)


if __name__ == '__main__':
    from parse_funcs.parse_derivation import parse_derivations
    from build_labels.translate_format import translate_format

    derivation_text_example = \
        ('intent:get_sunrise ( [ location: intent:get_location ( [ location_user: null ] [ search_radius: null ] '
         '[ location_modifier: London ] ) ] [ date_time: Next*spaceFriday ] [ weather: Rainy ] )')
    dataset_type = 'topv2'
    al_exp = parse_derivations(derivation_text_example, dataset_type)
    gen_labels = translate_format([al_exp], dataset_name=dataset_type)
    print(generate_nl(gen_labels, dataset_name=dataset_type)[0])
