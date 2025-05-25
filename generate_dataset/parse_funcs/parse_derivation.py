from ._parse_derivation_topv2 import _parse_derivation_topv2, _parse_derivation_conic10k
from generate_dataset.modeling.base_classes import Assertion, Formula


_FACT_TYPE = Assertion | Formula
parse_funcs_dict = {'topv2': _parse_derivation_topv2,
                    'conic10k': _parse_derivation_conic10k}


def parse_derivations(derivation_texts: list[str] | str, dataset_name: str) -> _FACT_TYPE | list[_FACT_TYPE]:
    parse_func = parse_funcs_dict[dataset_name]
    if isinstance(derivation_texts, str):
        return parse_func(derivation_texts)

    return [parse_func(d) for d in derivation_texts if d]  # 可能有不合法的情况，从而为None


if __name__ == '__main__':
    # 字符串 -> class
    derivation_text_example = \
        'intent:get_sunrise ( [ location: London ] [ date_time: Next*spaceFriday ] [ weather: Rainy ] )'
    dataset = 'topv2'
    print(parse_derivations(derivation_text_example, dataset))
