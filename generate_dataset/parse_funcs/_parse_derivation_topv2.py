from parse_funcs.base_classes import Assertion, Formula, BaseOperator, Term, BaseIndividual
from ._co_namespace import Declared_Operators


# todo: 这里应当加层校验机制，让它和对应rule的cfg文件保持一致，或者就直接从cfg文件中导出
get_weather = BaseOperator(
    name="GET_WEATHER",
    input_type=["DATE_TIME", "WEATHER_TEMPERATURE_UNIT", "LOCATION", "WEATHER_ATTRIBUTE"],
    output_type="WEATHER",
    description="get weather condition"
)

get_sunset = BaseOperator(
    name="GET_SUNSET",
    input_type=["LOCATION", "DATE_TIME"],
    output_type="sunset_time",
    description="get sunset time"
)

get_sunrise = BaseOperator(
    name="GET_SUNRISE",
    input_type=["LOCATION", "DATE_TIME"],
    output_type="sunrise_time",
    description="get sunrise time"
)

get_location = BaseOperator(
    name="GET_LOCATION",
    input_type=["LOCATION_USER"],
    output_type="LOCATION",
    description="get location"
)

is_growth_enterprise = BaseOperator(
    name="is_growth_enterprise",
    input_type=["COMPANY_NAME", "DATE_TIME"],
    output_type="Determine whether the company is a growth enterprise",
    description="Determine whether the company is a growth enterprise"
)

industry_development_trend = BaseOperator(
    name="industry_development_trend",
    input_type=["INDUSTRY_NAME", "DATE_TIME"],
    output_type="Industry Development Trends",
    description="Industry Development Trends"
)
get_company_competitors = BaseOperator(
    name="get_company_competitors",
    input_type=["COMPANY_NAME", "DATE_TIME"],
    output_type="company's competitors",
    description="company's competitors"
)

get_raw_material_price_and_ratio = BaseOperator(
    name="get_raw_material_price_and_ratio",
    input_type=["PRODUCT_NAME", "DATE_TIME"],
    output_type="Raw materials and their proportion for obtaining the product",
    description="Raw materials and their proportion for obtaining the product"
)
get_industry_chain_product = BaseOperator(
    name="get_industry_chain_product",
    input_type=["INDUSTRY_NAME", "DATE_TIME"],
    output_type="main chain products in the industry",
    description="main chain products in the industry"
)
get_company_profitability = BaseOperator(
    name="get_company_profitability",
    input_type=["COMPANY_NAME", "DATE_TIME"],
    output_type="company profitability",
    description="company profitability"
)

get_financial_metric = BaseOperator(
    name="get_financial_metric",
    input_type=["METRIC", "COMPANY_NAME", "DATE_TIME"],
    output_type="obtain financial indicator values",
    description="obtain financial indicator values"
)

get_financial_metric_change_situation = BaseOperator(
    name="get_financial_metric_change_situation",
    input_type=["METRIC", "COMPANY_NAME", "BEGIN_DATE", "END_DATE"],
    output_type="obtain changes in financial indicator values",
    description="obtain changes in financial indicator values"
)

get_financial_metric_change_rate = BaseOperator(
    name="get_financial_metric_change_rate",
    input_type=["METRIC", "COMPANY_NAME", "BEGIN_DATE", "END_DATE"],
    output_type="Obtain the rate of change of financial indicator values",
    description="Obtain the rate of change of financial indicator values"
)

get_exchange_rate = BaseOperator(
    name="get_exchange_rate",
    input_type=["ORIGINAL_CURRENCY_NAME", "TARGET_CURRENCY_NAME"],
    output_type="Obtain exchange rate information",
    description="Obtain exchange rate information"
)

get_cookbook = BaseOperator(
    name="get_cookbook",
    input_type=["FOOD_NAME", "MAX_CALORIES_NUM", "COPIES_NUM"],
    output_type="Get recipe information",
    description="Get recipe information"
)

get_movies = BaseOperator(
    name="get_movies",
    input_type=["MOVIE_KEYWORD", "DATE_TIME"],
    output_type="Get detailed information about the movie",
    description="Get detailed information about the movie"
)

get_news = BaseOperator(
    name="get_news",
    input_type=["QUERY_KEYWORD", "DATE_TIME", "LOCATION"],
    output_type="news information",
    description="news information"
)

get_stock_price = BaseOperator(
    name="get_stock_price",
    input_type=["COMPANY_NAME", "COMPANY_CODE"],
    output_type="stock price information",
    description="stock price information"
)
#
get_event = BaseOperator(
    name="GET_EVENT",
    input_type=["CATEGORY_EVENT", "DATE_TIME", "LOCATION", "ATTRIBUTE_EVENT", "NAME_EVENT", "ORDINAL", "ORGANIZER_EVENT"],
    output_type="event or activity information",
    description="get event or activity information\t[SL:CATEGORY_EVENT: event category\t[SL:DATE_TIME: event occurrence time\t[SL:LOCATION: event occurrence location"
                "\t[SL:ATTRIBUTE_EVENT: some attribute with the event\t[SL:NAME_EVENT: event name\t[SL:ORDINAL: the ordinal number of the selected reminder or pointing towards the future"
)

create_reminder = BaseOperator(
    name="CREATE_REMINDER",
    input_type=["PERSON_REMINDED", "REMINDER_DATE_TIME", "TODO", "RECURRING_DATE_TIME"],
    output_type="create reminder",
    description="create reminder.\t[SL:PERSON_REMINDED: the reminded target\t[SL:REMINDER_DATE_TIME: reminder date time\t[SL:TODO: reminder of things to do\t[SL:ORDINAL: reminder of things to do\t[SL:RECURRING_DATE_TIME: number and duration of recurring"
)

delete_reminder = BaseOperator(
    name="DELETE_REMINDER",
    input_type=["PERSON_REMINDED", "REMINDER_DATE_TIME", "TODO", "AMOUNT", "ORDINAL", "RECURRING_DATE_TIME"],
    output_type="delete reminder",
    description="delete previously set reminders.\t[SL:PERSON_REMINDED: the reminded target\t[SL:REMINDER_DATE_TIME: reminder date time\t[SL:TODO: reminder of things to do\t[SL:AMOUNT: number of selected or pointed reminders\t[SL:ORDINAL: the ordinal number of the selected reminder\t[SL:RECURRING_DATE_TIME: number and duration of recurring"
)

get_reminder = BaseOperator(
    name="GET_REMINDER",
    input_type=["PERSON_REMINDED", "REMINDER_DATE_TIME", "TODO", "AMOUNT", "ORDINAL", "METHOD_RETRIEVAL_REMINDER"],
    output_type="get reminder information",
    description="get previously set reminder content.\t[SL:PERSON_REMINDED: the reminded target\t[SL:REMINDER_DATE_TIME: reminder date time\t[SL:TODO: reminder of things to do\t[SL:AMOUNT: number of selected or pointed reminders\t[SL:ORDINAL: the ordinal number of the selected reminder\t[SL:METHOD_RETRIEVAL_REMINDER: Reminder method"
)

get_reminder_amount = BaseOperator(
    name="GET_REMINDER_AMOUNT",
    input_type=["PERSON_REMINDED", "REMINDER_DATE_TIME", "TODO"],
    output_type="get reminder amount",
    description="get number of times the reminder has been set\t[SL:PERSON_REMINDED: the reminded target\t[SL:REMINDER_DATE_TIME: reminder date time\t[SL:TODO: reminder of things to do"
)

get_reminder_date_time = BaseOperator(
    name="GET_REMINDER_DATE_TIME",
    input_type=["PERSON_REMINDED", "DATE_TIME", "TODO", "AMOUNT", "ORDINAL"],
    output_type="get reminder date time",
    description="get the reminder time for the previously set reminder\t[SL:PERSON_REMINDED: the reminded target\t[SL:DATE_TIME: current date time\t[SL:TODO: reminder of things to do\t[SL:AMOUNT: number of selected or pointed reminders\t[SL:ORDINAL: the ordinal number of the selected reminder"
)

get_reminder_location = BaseOperator(
    name="GET_REMINDER_LOCATION",
    input_type=["PERSON_REMINDED", "REMINDER_DATE_TIME", "TODO", "ORDINAL", "METHOD_RETRIEVAL_REMINDER"],
    output_type="get reminder location",
    description="get the event sending location of the previously set reminder\t[SL:PERSON_REMINDED: the reminded target\t[SL:REMINDER_DATE_TIME: reminder date time\t[SL:TODO: reminder of things to do\t[SL:ORDINAL: the ordinal number of the selected reminder\t[SL:METHOD_RETRIEVAL_REMINDER: Reminder method"
)

update_reminder_todo = BaseOperator(
    name="UPDATE_REMINDER_TODO",
    input_type=["PERSON_REMINDED", "REMINDER_DATE_TIME", "TODO", "TODO_NEW", "RECURRING_DATE_TIME"],
    output_type="update reminder todo",
    description="Update the reminder memory from Todo to the new Todo\t[SL:PERSON_REMINDED: the reminded target\t[SL:REMINDER_DATE_TIME: reminder date time\t[SL:TODO: reminder content set in the past\t[SL:TODO_NEW: Updated reminder content\t[SL:RECURRING_DATE_TIME: number and duration of recurring"
)

get_todo = BaseOperator(
    name="GET_TODO",
    input_type=["TODO_ANOTHER", "TODO_DATE_TIME", "ATTENDEE"],
    output_type="get todo information",
    description="get todo information\t[SL:TODO_ANOTHER: reminder of things to do\t[SL:TODO_DATE_TIME:todo occurrence time\t[SL:ATTENDEE:todo occurrence time\t[SL:ATTENDEE:attendee"
)

get_contact = BaseOperator(
    name="GET_CONTACT",
    input_type=["CONTACT_RELATED", "TYPE_RELATION", "CONTACT"],
    output_type="get contact",
    description="get relationship with people\t[SL:TYPE_RELATION: interpersonal relationship\t[SL:CONTACT: contacts"
)

get_recurring_date_time = BaseOperator(
    name="GET_RECURRING_DATE_TIME",
    input_type=["FREQUENCY", "DATE_TIME"],
    output_type="number and duration of recurring",
    description = "number and duration of recurring\t[SL:FREQUENCY: cycle frequency\t[SL:DATE_TIME: cycle time period"
)

dummy_operator = BaseOperator(
    name="dummy_operator",
    input_type=[],
    output_type='',
    description=''
)


Declared_Operators.update({
    "GET_WEATHER": get_weather,
    "GET_SUNSET": get_sunset,
    "GET_SUNRISE": get_sunrise,
    "GET_LOCATION": get_location,
    "GET_EVENT": get_event,
    "is_growth_enterprise": is_growth_enterprise,
    "industry_development_trend": industry_development_trend,
    "get_company_competitors": get_company_competitors,
    "get_raw_material_price_and_ratio": get_raw_material_price_and_ratio,
    "get_industry_chain_product": get_industry_chain_product,
    "get_company_profitability": get_company_profitability,
    "get_financial_metric": get_financial_metric,
    "get_financial_metric_change_situation": get_financial_metric_change_situation,
    "get_financial_metric_change_rate": get_financial_metric_change_rate,
    "get_exchange_rate": get_exchange_rate,
    "get_cookbook": get_cookbook,
    "get_movies": get_movies,
    "get_news": get_news,
    "get_stock_price": get_stock_price,
    "CREATE_REMINDER": create_reminder,
    "DELETE_REMINDER": delete_reminder,
    "GET_REMINDER": get_reminder,
    "GET_REMINDER_AMOUNT": get_reminder_amount,
    "GET_REMINDER_DATE_TIME": get_reminder_date_time,
    "GET_REMINDER_LOCATION": get_reminder_location,
    "UPDATE_REMINDER_TODO": update_reminder_todo,
    "GET_TODO": get_todo,
    "GET_CONTACT": get_contact,
    "GET_RECURRING_DATE_TIME": get_recurring_date_time,
    "dummy_operator": dummy_operator
})


def __clean_text(text: str) -> str:
    """
    将一些特殊的转义符号等退回正常的格式。目前先认为仅对variables使用即可（这个是随着parse_derivation自定义的。top的规则里我们不对
    intent使用特殊字符
    """
    text = text.replace('*space', ' ')
    return text


def __split_operator_variables(derivation_text: str) -> tuple[str, str]:
    """
    Splits the derivation text into the operator name and the variables text.
    Example: "get_sunset([LOCATION:null][DATE_TIME:get_next_day(...)][weather:Rainy])"
    returns ("get_sunset", "[LOCATION:null][DATE_TIME:get_next_day(...)][weather:Rainy]")
    """
    start = derivation_text.find("(")
    end = derivation_text.rfind(")")
    if start == -1 or end == -1:
        raise ValueError("Invalid derivation text format")
    operator_name = derivation_text[:start][len('intent:'):].strip()
    variables_text = derivation_text[start + 1:end].strip()
    return operator_name, variables_text


def __extract_variables(variables_text: str, operator: BaseOperator) -> list[BaseIndividual | Term]:
    """
    Extracts variables from the variables text and returns them as a list of BaseIndividual or Term objects.
    Example: "[LOCATION:null][DATE_TIME:get_next_day(...)][weather:Rainy]"
    returns [BaseIndividual("LOCATION", null), Term(get_next_day, [...]), BaseIndividual("WEATHER", "Rainy")]
    """
    def find_matching_bracket(text: str, start: int) -> int:
        """
        Finds the index of the matching closing bracket for a nested structure.
        """
        print("text")
        print(text)

        stack = 1  # The opening bracket at 'start' is already found
        for i in range(start + 1, len(text)):
            if text[i] == '[':
                stack += 1
            elif text[i] == ']':
                stack -= 1
                if stack == 0:
                    return i
        raise ValueError("Mismatched brackets in variables text")

    variables = []
    i = 0
    while i < len(variables_text):
        if variables_text[i] == '[':
            # Find the closing bracket for this variable
            end = find_matching_bracket(variables_text, i)
            inner_text = variables_text[i + 1:end]  # Strip the outer brackets
            name, value = inner_text.split(":", 1)

            assert name in operator.inputType + [operator.outputType], \
                f"Name {name} not found in input types {operator.inputType} or output type {operator.outputType}"

            if value.startswith("intent:"):
                sub_operator_name, sub_variables_text = __split_operator_variables(value)
                sub_operator = Declared_Operators[sub_operator_name]
                sub_variables = __extract_variables(sub_variables_text, sub_operator)
                variables.append(Term(operator=sub_operator, variables=sub_variables))
            else:
                value = __clean_text(value)
                variables.append(BaseIndividual(value=value))

            i = end + 1
        else:
            i += 1

    return variables


def _parse_derivation_topv2(derivation_text: str) -> Assertion:
    """
    把生成的表达式转为topv2的格式
    get_sunset ( [ LOCATION: null ] [ DATE_TIME: get_next_day ( [ datetime: This*spaceMonday ] ) ] [ weather: Rainy ] )
    """
    derivation_text = derivation_text.replace(' ', '')
    operator_name, variables_text = __split_operator_variables(derivation_text)
    operator = Declared_Operators[operator_name]
    variables = __extract_variables(variables_text, operator)
    term_lhs = Term(operator=operator,
                    variables=variables)
    term_rhs = Term(Declared_Operators['dummy_operator'],
                    variables=[])

    return Assertion(lhs=term_lhs, rhs=term_rhs)


if __name__ == '__main__':
    derivation_text_example = \
        'intent:GET_EVENT ( [ CATEGORY_EVENT: null ] [ DATE_TIME: later*spacetoday ] [ LOCATION: null ] [ ATTRIBUTE_EVENT: families ] [ NAME_EVENT: null ] [ ORDINAL: null ] [ ORGANIZER_EVENT: null ] )'
    print(_parse_derivation_topv2(derivation_text_example))
