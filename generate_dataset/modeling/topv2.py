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

Coordinate = BaseOperator(
    name="Coordinate",
    input_type=["POINT"],
    output_type=["NUMBER", "NUMBER"],
    description="Get the (x, y) coordinates of a point"
)

XCoordinate = BaseOperator(
    name="XCoordinate",
    input_type=["POINT"],
    output_type="NUMBER",
    description="Get the x-coordinate of a point"
)

YCoordinate = BaseOperator(
    name="YCoordinate",
    input_type=["POINT"],
    output_type="NUMBER",
    description="Get the y-coordinate of a point"
)

LocusEquation = BaseOperator(
    name="LocusEquation",
    input_type=["POINT"],
    output_type="EXPRESSION",
    description="Get the equation of a point's locus"
)

Locus = BaseOperator(
    name="Locus",
    input_type=["POINT"],
    output_type="CURVE",
    description="Get the curve type representing a point's locus"
)

Quadrant = BaseOperator(
    name="Quadrant",
    input_type=["POINT"],
    output_type="NUMBER",
    description="Determine which quadrant a point is in"
)

Focus = BaseOperator(
    name="Focus",
    input_type=["CONICSECTION"],
    output_type=["POINT", "POINT"],
    description="Get the foci of a conic section"
)

RightFocus = BaseOperator(
    name="RightFocus",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the right focus of a conic section"
)

LeftFocus = BaseOperator(
    name="LeftFocus",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the left focus of a conic section"
)

LowerFocus = BaseOperator(
    name="LowerFocus",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the lower focus of a conic section"
)

UpperFocus = BaseOperator(
    name="UpperFocus",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the upper focus of a conic section"
)

FocalLength = BaseOperator(
    name="FocalLength",
    input_type=["CONICSECTION"],
    output_type="NUMBER",
    description="Get the focal length of a conic section"
)

HalfFocalLength = BaseOperator(
    name="HalfFocalLength",
    input_type=["CONICSECTION"],
    output_type="NUMBER",
    description="Get half of the focal length of a conic section"
)

SymmetryAxis = BaseOperator(
    name="SymmetryAxis",
    input_type=["CONICSECTION"],
    output_type="AXIS",
    description="Get the symmetry axis of a conic section"
)

Eccentricity = BaseOperator(
    name="Eccentricity",
    input_type=["CONICSECTION"],
    output_type="NUMBER",
    description="Get the eccentricity of a conic section"
)

Vertex = BaseOperator(
    name="Vertex",
    input_type=["CONICSECTION"],
    output_type=["POINT", "POINT"],
    description="Get the vertices of a conic section"
)

UpperVertex = BaseOperator(
    name="UpperVertex",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the upper vertex of a conic section"
)

LowerVertex = BaseOperator(
    name="LowerVertex",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the lower vertex of a conic section"
)

LeftVertex = BaseOperator(
    name="LeftVertex",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the left vertex of a conic section"
)

RightVertex = BaseOperator(
    name="RightVertex",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the right vertex of a conic section"
)

Directrix = BaseOperator(
    name="Directrix",
    input_type=["CONICSECTION"],
    output_type=["LINE", "LINE"],
    description="Get the directrices of a conic section"
)

LeftDirectrix = BaseOperator(
    name="LeftDirectrix",
    input_type=["CONICSECTION"],
    output_type="LINE",
    description="Get the left directrix of a conic section"
)

RightDirectrix = BaseOperator(
    name="RightDirectrix",
    input_type=["CONICSECTION"],
    output_type="LINE",
    description="Get the right directrix of a conic section"
)

MajorAxis = BaseOperator(
    name="MajorAxis",
    input_type=["ELLIPSE"],
    output_type="LINESEGMENT",
    description="Get the major axis of an ellipse"
)

MinorAxis = BaseOperator(
    name="MinorAxis",
    input_type=["ELLIPSE"],
    output_type="LINESEGMENT",
    description="Get the minor axis of an ellipse"
)

RealAxis = BaseOperator(
    name="RealAxis",
    input_type=["ELLIPSE"],
    output_type="LINESEGMENT",
    description="Get the real axis of an ellipse"
)

ImaginaryAxis = BaseOperator(
    name="ImaginaryAxis",
    input_type=["ELLIPSE"],
    output_type="LINESEGMENT",
    description="Get the imaginary axis of an ellipse"
)

LeftPart = BaseOperator(
    name="LeftPart",
    input_type=["HYPERBOLA"],
    output_type="CURVE",
    description="Get the left part of a hyperbola"
)

RightPart = BaseOperator(
    name="RightPart",
    input_type=["HYPERBOLA"],
    output_type="CURVE",
    description="Get the right part of a hyperbola"
)

Asymptote = BaseOperator(
    name="Asymptote",
    input_type=["HYPERBOLA"],
    output_type=["LINE", "LINE"],
    description="Get the asymptotes of a hyperbola"
)

Diameter = BaseOperator(
    name="Diameter",
    input_type=["CIRCLE"],
    output_type="NUMBER",
    description="Get the diameter of a circle"
)

Radius = BaseOperator(
    name="Radius",
    input_type=["CIRCLE"],
    output_type="NUMBER",
    description="Get the radius of a circle"
)

Perimeter = BaseOperator(
    name="Perimeter",
    input_type=["CONICSECTION"],
    output_type="NUMBER",
    description="Get the perimeter of a conic section"
)

Center = BaseOperator(
    name="Center",
    input_type=["CONICSECTION"],
    output_type="POINT",
    description="Get the center of a conic section"
)

Expression = BaseOperator(
    name="Expression",
    input_type=["CURVE"],
    output_type="EXPRESSION",
    description="Get the expression of a curve"
)

LineOf = BaseOperator(
    name="LineOf",
    input_type=["POINT", "POINT"],
    output_type="LINE",
    description="Get the line passing through two points"
)

Slope = BaseOperator(
    name="Slope",
    input_type=["LINE"],
    output_type="NUMBER",
    description="Get the slope of a line"
)

Inclination = BaseOperator(
    name="Inclination",
    input_type=["LINE"],
    output_type="NUMBER",
    description="Get the inclination angle of a line"
)

Intercept = BaseOperator(
    name="Intercept",
    input_type=["LINE", "AXIS"],
    output_type="NUMBER",
    description="Get the intercept of a line on an axis"
)

LineSegmentOf = BaseOperator(
    name="LineSegmentOf",
    input_type=["POINT", "POINT"],
    output_type="LINESEGMENT",
    description="Get the line segment between two points"
)

Length = BaseOperator(
    name="Length",
    input_type=["LINESEGMENT"],
    output_type="NUMBER",
    description="Get the length of a line segment"
)

MidPoint = BaseOperator(
    name="MidPoint",
    input_type=["LINESEGMENT"],
    output_type="POINT",
    description="Get the midpoint of a line segment"
)

OverlappingLine = BaseOperator(
    name="OverlappingLine",
    input_type=["LINESEGMENT"],
    output_type="LINE",
    description="Get the line that overlaps with a line segment"
)

PerpendicularBisector = BaseOperator(
    name="PerpendicularBisector",
    input_type=["LINESEGMENT"],
    output_type="LINE",
    description="Get the perpendicular bisector of a line segment"
)

Endpoint = BaseOperator(
    name="Endpoint",
    input_type=["LINESEGMENT"],
    output_type=["POINT", "POINT"],
    description="Get the endpoints of a line segment"
)

Projection = BaseOperator(
    name="Projection",
    input_type=["POINT", "LINE"],
    output_type="POINT",
    description="Get the projection of a point onto a line"
)

InterceptChord = BaseOperator(
    name="InterceptChord",
    input_type=["LINE", "CONICSECTION"],
    output_type="LINESEGMENT",
    description="Get the chord intercepted by a line on a conic section"
)

TriangleOf = BaseOperator(
    name="TriangleOf",
    input_type=["POINT", "POINT", "POINT"],
    output_type="TRIANGLE",
    description="Get the triangle formed by three points"
)

InscribedCircle = BaseOperator(
    name="InscribedCircle",
    input_type=["TRIANGLE"],
    output_type="CIRCLE",
    description="Get the inscribed circle of a triangle"
)

CircumCircle = BaseOperator(
    name="CircumCircle",
    input_type=["TRIANGLE"],
    output_type="CIRCLE",
    description="Get the circumcircle of a triangle"
)

Incenter = BaseOperator(
    name="Incenter",
    input_type=["TRIANGLE"],
    output_type="POINT",
    description="Get the incenter of a triangle"
)

Orthocenter = BaseOperator(
    name="Orthocenter",
    input_type=["TRIANGLE"],
    output_type="POINT",
    description="Get the orthocenter of a triangle"
)

Circumcenter = BaseOperator(
    name="Circumcenter",
    input_type=["TRIANGLE"],
    output_type="POINT",
    description="Get the circumcenter of a triangle"
)

AngleOf = BaseOperator(
    name="AngleOf",
    input_type=["POINT", "POINT", "POINT"],
    output_type="DEGREE",
    description="Get the angle formed by three points"
)

VectorOf = BaseOperator(
    name="VectorOf",
    input_type=["POINT", "POINT"],
    output_type="VECTOR",
    description="Get the vector from one point to another"
)

DotProduct = BaseOperator(
    name="DotProduct",
    input_type=["VECTOR", "VECTOR"],
    output_type="NUMBER",
    description="Calculate the dot product of two vectors"
)

NumIntersection = BaseOperator(
    name="NumIntersection",
    input_type=["CURVE", "CURVE"],
    output_type="NUMBER",
    description="Get the number of intersection points between two curves"
)

Intersection = BaseOperator(
    name="Intersection",
    input_type=["CURVE", "CURVE"],
    output_type=["POINT", "POINT"],
    description="Get the intersection points of two curves"
)

FootPoint = BaseOperator(
    name="FootPoint",
    input_type=["LINE", "LINE"],
    output_type="POINT",
    description="Get the foot point from one line to another"
)

Distance = BaseOperator(
    name="Distance",
    input_type=["POINT", "POINT"],
    output_type="NUMBER",
    description="Calculate the distance between two points"
)

Area = BaseOperator(
    name="Area",
    input_type=["CONICSECTION"],
    output_type="NUMBER",
    description="Calculate the area of a conic section"
)

TangentPoint = BaseOperator(
    name="TangentPoint",
    input_type=["LINE", "CIRCLE"],
    output_type="POINT",
    description="Get the tangent point between a line and a circle"
)

TangentOfPoint = BaseOperator(
    name="TangentOfPoint",
    input_type=["POINT", "CURVE"],
    output_type="LINE",
    description="Get the tangent line at a point on a curve"
)

TangentOnPoint = BaseOperator(
    name="TangentOnPoint",
    input_type=["POINT", "CURVE"],
    output_type="LINE",
    description="Get the tangent line on a point of a curve"
)

Abs = BaseOperator(
    name="Abs",
    input_type=["LINESEGMENT"],
    output_type="NUMBER",
    description="Get the absolute value (length) of a line segment"
)

lg = BaseOperator(
    name="lg",
    input_type=["NUMBER"],
    output_type="NUMBER",
    description="Calculate the logarithm of a number"
)

InterReciprocal = BaseOperator(
    name="InterReciprocal",
    input_type=["NUMBER"],
    output_type="NUMBER",
    description="Calculate the reciprocal of a number"
)

sqrt = BaseOperator(
    name="sqrt",
    input_type=["NUMBER"],
    output_type="NUMBER",
    description="Calculate the square root of a number"
)

Max = BaseOperator(
    name="Max",
    input_type=["NUMBER"],
    output_type="NUMBER",
    description="Get the maximum value"
)

Min = BaseOperator(
    name="Min",
    input_type=["NUMBER"],
    output_type="NUMBER",
    description="Get the minimum value"
)

ApplyUnit = BaseOperator(
    name="ApplyUnit",
    input_type=["NUMBER", "DEGREESTR"],
    output_type=None,
    description="Apply a unit to a number"
)

Between = BaseOperator(
    name="Between",
    input_type=["POINT", "POINT", "POINT"],
    output_type="BOOL",
    description="Check if a point is between two other points"
)

IsIntersect = BaseOperator(
    name="IsIntersect",
    input_type=["CURVE", "CURVE"],
    output_type="BOOL",
    description="Check if two curves intersect"
)

IsPerpendicular = BaseOperator(
    name="IsPerpendicular",
    input_type=["LINE", "LINE"],
    output_type="BOOL",
    description="Check if two lines are perpendicular"
)

IsParallel = BaseOperator(
    name="IsParallel",
    input_type=["LINE", "LINE"],
    output_type="BOOL",
    description="Check if two lines are parallel"
)

IsDiameter = BaseOperator(
    name="IsDiameter",
    input_type=["LINESEGMENT", "CIRCLE"],
    output_type="BOOL",
    description="Check if a line segment is a diameter of a circle"
)

IsChordOf = BaseOperator(
    name="IsChordOf",
    input_type=["LINESEGMENT", "CONICSECTION"],
    output_type="BOOL",
    description="Check if a line segment is a chord of a conic section"
)

IsTangent = BaseOperator(
    name="IsTangent",
    input_type=["LINE", "CONICSECTION"],
    output_type="BOOL",
    description="Check if a line is tangent to a conic section"
)

IsInscribedCircle = BaseOperator(
    name="IsInscribedCircle",
    input_type=["CIRCLE", "TRIANGLE"],
    output_type="BOOL",
    description="Check if a circle is the inscribed circle of a triangle"
)

IsCircumCircle = BaseOperator(
    name="IsCircumCircle",
    input_type=["CIRCLE", "TRIANGLE"],
    output_type="BOOL",
    description="Check if a circle is the circumcircle of a triangle"
)

IsInTangent = BaseOperator(
    name="IsInTangent",
    input_type=["CIRCLE", "CIRCLE"],
    output_type="BOOL",
    description="Check if two circles are internally tangent"
)

IsOutTangent = BaseOperator(
    name="IsOutTangent",
    input_type=["CIRCLE", "CIRCLE"],
    output_type="BOOL",
    description="Check if two circles are externally tangent"
)

PointOnCurve = BaseOperator(
    name="PointOnCurve",
    input_type=["POINT", "CURVE"],
    output_type="BOOL",
    description="Check if a point lies on a curve"
)

And = BaseOperator(
    name="And",
    input_type=["ASSERTION", "ASSERTION"],
    output_type="BOOL",
    description="Logical AND operation between two assertions"
)

Negation = BaseOperator(
    name="Negation",
    input_type=["ASSERTION"],
    output_type="BOOL",
    description="Logical negation of an assertion"
)

all_operators = {
    "Coordinate": Coordinate,
    "XCoordinate": XCoordinate,
    "YCoordinate": YCoordinate,
    "LocusEquation": LocusEquation,
    "Locus": Locus,
    "Quadrant": Quadrant,
    "Focus": Focus,
    "RightFocus": RightFocus,
    "LeftFocus": LeftFocus,
    "LowerFocus": LowerFocus,
    "UpperFocus": UpperFocus,
    "FocalLength": FocalLength,
    "HalfFocalLength": HalfFocalLength,
    "SymmetryAxis": SymmetryAxis,
    "Eccentricity": Eccentricity,
    "Vertex": Vertex,
    "UpperVertex": UpperVertex,
    "LowerVertex": LowerVertex,
    "LeftVertex": LeftVertex,
    "RightVertex": RightVertex,
    "Directrix": Directrix,
    "LeftDirectrix": LeftDirectrix,
    "RightDirectrix": RightDirectrix,
    "MajorAxis": MajorAxis,
    "MinorAxis": MinorAxis,
    "RealAxis": RealAxis,
    "ImaginaryAxis": ImaginaryAxis,
    "LeftPart": LeftPart,
    "RightPart": RightPart,
    "Asymptote": Asymptote,
    "Diameter": Diameter,
    "Radius": Radius,
    "Perimeter": Perimeter,
    "Center": Center,
    "Expression": Expression,
    "LineOf": LineOf,
    "Slope": Slope,
    "Inclination": Inclination,
    "Intercept": Intercept,
    "LineSegmentOf": LineSegmentOf,
    "Length": Length,
    "MidPoint": MidPoint,
    "OverlappingLine": OverlappingLine,
    "PerpendicularBisector": PerpendicularBisector,
    "Endpoint": Endpoint,
    "Projection": Projection,
    "InterceptChord": InterceptChord,
    "TriangleOf": TriangleOf,
    "InscribedCircle": IsInscribedCircle,
    "CircumCircle": CircumCircle,
    "Incenter": Incenter,
    "Orthocenter": Orthocenter,
    "Circumcenter": Circumcenter,
    "AngleOf": AngleOf,
    "VectorOf": VectorOf,
    "DotProduct": DotProduct,
    "NumIntersection": NumIntersection,
    "Intersection": Intersection,
    "FootPoint": FootPoint,
    "Distance": Distance,
    "Area": Area,
    "TangentPoint": TangentPoint,
    "TangentOfPoint": TangentOfPoint,
    "TangentOnPoint": TangentOnPoint,
    "Abs": Abs,
    "lg": lg,
    "InterReciprocal": InterReciprocal,
    "sqrt": sqrt,
    "Max": Max,
    "Min": Min,
    "ApplyUnit": ApplyUnit,
    "Between": Between,
    "IsIntersect": IsIntersect,
    "IsPerpendicular": IsPerpendicular,
    "IsParallel": IsParallel,
    "IsDiameter": IsDiameter,
    "IsChordOf": IsChordOf,
    "IsTangent": IsTangent,
    "IsInscribedCircle": IsInscribedCircle,
    "IsCircumCircle": IsCircumCircle,
    "IsInTangent": IsInTangent,
    "IsOutTangent": IsOutTangent,
    "PointOnCurve": PointOnCurve,
    "And": And,
    "Negation": Negation,
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

# all_operators = {
#     "GET_WEATHER": get_weather,
#     "GET_SUNSET": get_sunset,
#     "GET_SUNRISE": get_sunrise,
#     "GET_LOCATION": get_location,
#     "GET_EVENT": get_event,
#     "is_growth_enterprise": is_growth_enterprise,
#     "industry_development_trend": industry_development_trend,
#     "get_company_competitors": get_company_competitors,
#     "get_raw_material_price_and_ratio": get_raw_material_price_and_ratio,
#     "get_industry_chain_product": get_industry_chain_product,
#     "get_company_profitability": get_company_profitability,
#     "get_financial_metric": get_financial_metric,
#     "get_financial_metric_change_situation": get_financial_metric_change_situation,
#     "get_financial_metric_change_rate": get_financial_metric_change_rate,
#     "get_exchange_rate": get_exchange_rate,
#     "get_cookbook": get_cookbook,
#     "get_movies": get_movies,
#     "get_news": get_news,
#     "get_stock_price": get_stock_price,
#     "CREATE_REMINDER": create_reminder,
#     "DELETE_REMINDER": delete_reminder,
#     "GET_REMINDER": get_reminder,
#     "GET_REMINDER_AMOUNT": get_reminder_amount,
#     "GET_REMINDER_DATE_TIME": get_reminder_date_time,
#     "GET_REMINDER_LOCATION": get_reminder_location,
#     "UPDATE_REMINDER_TODO": update_reminder_todo,
#     "GET_TODO": get_todo,
#     "GET_CONTACT": get_contact,
#     "GET_RECURRING_DATE_TIME": get_recurring_date_time,
#     "dummy_operator": dummy_operator
# }
