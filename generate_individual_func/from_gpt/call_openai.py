instruct = '''
You are an instantiation generator, when l give you some concepts about weather like DATE_TIME, WEATHER_TEMPERATURE_UNIT, LOCATION, WEATHER_ATTRIBUTE, etc. you need to give a detailed English example about it.
The WEATHER_ATTRIBUTES represent weather conditions such as rain, snow, sunny, hot and so on. WEATHER_TEMPERATURE_UNIT represents a unit of temperature, such as degrees Celsius, Fahrenheit, and so on. DATE_TlME indicates the time that want to get the weather, which can be vague like "tonight, tomorrow" or precise like "this Monday, next Friday, March 5" and so on. UNSUPPORTED_WEATHER means that there is a preliminary guess about the weather condition and then asks if the guess is correct.
Don't give the process, just return one and only one instance.

######### examples ###########
Question: Please randomly give an instance about "DATE TIME"
Answer: Tonight

Question: Please randomly give an instance about "LOCATION"
Answer: sydney

Question: Please randomly give an instance about "WEATHER_ATTRIBUTE"
Answer: sunny

Question: Please randomly give an instance about "WEATHER_TEMPERATURE_UNIT"
Answer: Celsius

'''
input = 'Question: Please randomly give an instance about "{}"\nAnswer:'

import os
import openai

# model param config
max_token_num=1000
summary_temperature = 0.8
model = "gpt-35-turbo"

api_version_num="2023-05-15"

openai.api_type ="azure"
openai.api_base =("")
openai.api_key = ""

os.environ["OPENAI_API_VERSION"]="2023-05-15"
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""

def generate_from_gpt(query):
    r = openai.ChatCompletion.create(
        engine = model,
        temperature = 0.8,
        api_version = api_version_num,
        messages = [
            {
                "role":"system",
                "content":instruct,
            },
            {
                "role":"user",
                "content":input.format(query),
            },
        ],
    )

    answer = r.choices[0]["message"]["content"]
    print(answer)
    return answer
