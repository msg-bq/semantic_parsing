# 之后你应该改成topv2_weather, topv2_event.....等
# todo: 这里应当加层校验机制，让它和对应rule的cfg文件保持一致，或者就直接从cfg文件中导出
# 应该反过来，先设置这个再导出cfg文件
from .base_classes import BaseOperator

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


all_operators = {
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
}
