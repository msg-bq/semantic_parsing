def _concept_instance_prompt(concept: str) -> str:
    instruct = '''You are an instantiation generator, when l give you a concept about weather like DATE_TIME, WEATHER_TEMPERATURE_UNIT, LOCATION, WEATHER_ATTRIBUTE, etc. you need to give some detailed example instances about it.
    The WEATHER_ATTRIBUTES represent weather conditions such as rain, snow, sunny, hot and so on. WEATHER_TEMPERATURE_UNIT represents a unit of temperature, such as degrees Celsius, Fahrenheit, and so on. DATE_TlME indicates the time that want to get the weather, which can be vague like "tonight, tomorrow" or precise like "this Monday, next Friday, March 5" and so on.
    Don't give the process or any extra words, just return 10 various instances. Again, no extra words and no extra punctuations.
    
    ######### examples ###########
    Question: Please randomly give an instance about "DATE TIME"
    Answer: [Tonight, 15 May, 2020.3.15, ...]
    
    Question: Please randomly give an instance about "LOCATION"
    Answer: [sydney, China, ...]
    
    Question: Please randomly give an instance about "WEATHER_ATTRIBUTE"
    Answer: [sunny, rainy, ...]
    
    Question: Please randomly give an instance about "WEATHER_TEMPERATURE_UNIT"
    Answer: [Celsius, ...]
    
    Question: Please randomly give an instance about "{}"\nAnswer:'''

    return instruct.format(concept)
