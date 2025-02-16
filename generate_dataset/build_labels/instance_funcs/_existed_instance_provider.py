import random
from mimesis import Generic, BaseProvider, BaseDataProvider
from mimesis.locales import Locale

from faker import Faker
from faker.providers import BaseProvider as FakerProvider


_CUSTOM_CONCEPTS_MAPPING = {'location': 'address'}  # 同义词的映射，比如虽然没有location函数，但不妨用address函数代替


class Weather_Temperatur_Provider(FakerProvider):
    def weather_temperature_unit(self) -> str:
        temperature_unit = ["c", "Celcius", "celcius", "Fahrenheit", "fahrenheit", "F", "f", "C"]
        return random.choice(temperature_unit)


def __get_existed_provider_from_mimesis(mimesis_generator) -> list[BaseProvider]:
    providers = []
    exclude = list(BaseProvider().__dict__.keys())
    # Exclude locale explicitly because
    # it is not a provider.
    exclude.append("locale")

    keys = list(mimesis_generator.__dict__.keys())
    for attr in keys:
        if attr not in exclude:
            if attr.startswith("_"):  # 这种的需要实例化，是locale-dependent的class
                providers.append(mimesis_generator.__getattr__(attr[1:]))
            else:  # 这种的直接取值，是locale-independent的class
                providers.append(getattr(mimesis_generator, attr))
    return providers


def __get_existed_data_func_from_mimesis_provider(provider: BaseProvider) -> dict[str, callable]:
    exclude = dir(BaseDataProvider)
    exclude.append('Meta')

    funcs = {}
    dct = type(provider).__dict__
    for fn in dct:
        if fn not in exclude and not fn.startswith('_'):
            fn_name = fn.replace('_', ' ')
            funcs[fn_name] = getattr(provider, fn)
    return funcs


def _get_existed_generator_mimesis(language=Locale.EN) -> dict[str, callable]:
    local_generator = Generic(locale=language)  # 指定语言为英文

    providers = __get_existed_provider_from_mimesis(local_generator)
    funcs = {}
    for p in providers:
        funcs.update(__get_existed_data_func_from_mimesis_provider(p))

    return funcs


def _get_existed_generator_faker(language='EN') -> dict[str, callable]:
    fake = Faker(locale=language)
    fake.add_provider(Weather_Temperatur_Provider)
    funcs = {fn.replace('_', ' '): fake.__getattr__(fn)
             for fn in dir(fake.factories[0]) if not fn.startswith('_')}  # 正常情况len(factories)就是==1的
    return funcs


def _get_existed_generators(language='EN') -> dict[str, callable]:
    if language == 'EN':
        lang_faker = 'EN'
        lang_mimesis = Locale.EN
    else:
        raise NotImplementedError

    funcs = {}
    funcs.update(_get_existed_generator_faker(lang_faker))
    funcs.update(_get_existed_generator_mimesis(lang_mimesis))

    global _CUSTOM_CONCEPTS_MAPPING
    for k, v in _CUSTOM_CONCEPTS_MAPPING.items():
        funcs[k] = funcs[v]

    return funcs


def _get_existed_generator_mimesis(language=Locale.EN) -> dict[str, callable]:
    local_generator = Generic(locale=language)  # 指定语言为英文

    providers = __get_existed_provider_from_mimesis(local_generator)
    funcs = {}
    for p in providers:
        funcs.update(__get_existed_data_func_from_mimesis_provider(p))

    return funcs