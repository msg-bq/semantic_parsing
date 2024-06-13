from concurrent.futures import ThreadPoolExecutor,as_completed
import json

##### instruct to generate naturalstatements in financial area ######
instruct1 = '''
You are a learned linguist, now please give some natural phrases to ask about the weather conditions.
First, you need to understand the semantics of the input expression. In parentheses are some of the things you need to include when asking about the weather. The WEATHER_ATTRIBUTES represent weather conditions such as rain, snow, sunny, hot and so on. WEATHER_TEMPERATURE_UNIT represents a unit of temperature, such as degrees Celsius, Fahrenheit, and so on. DATE_TlME indicates the time that want to get the weather, which can be vague like "tonight, tomorrow" or precise like "this Monday, next Friday, March 5" and so on. UNSUPPORTED_WEATHER means that there is a preliminary guess about the weather condition and then asks if the guess is correct.
Then, you should rewrite it into 3 rich natural sentences with different styles and grammatical rules, where:
The first natural statement can be asked in a direct and concise way, the second sentence can be asked in a different way while maintaining the same semantics and the third sentence can been riched by adding some additional statements appropriately.
It is important to note that these 3 sentences must contain all the information about the given expression, each sentence must have everything in parentheses and must keep the expression of the original expression!
Don't give the process, just return the final answer.
Answers should be returned in the following python dictionary format:
{'expression': "input expressions",
 'sentences': ["first sentence", "second sentence", "third sentence"]
}

######### input expression ##########
'''

######### instruct to generate naturalstatements in TOPv2 weather area #####
instruct2 = '''
You are a professional tagger, give you a natural statement about querying the weather conditions, and its corresponding intention like GET_WEATHER, GET_SUNSET, GET_SUNRISE, UNSUPPORTED_WEATHER etc. and its slots like DATE_TIME, WEATHER_TEMPERATURE_UNIT, LOCATION, WEATHER_ATTRIBUTE, etc.
The WEATHER_ATTRIBUTES represent weather conditions. WEATHER_TEMPERATURE_UNIT represents a unit of temperature. DATE_TlME indicates the time that want to get the weather.
You need to follow the order of natural statements, understand each word in turn, and make appropriate segmentation to match the corresponding slot for each part. Don't change the word order of natural sentences during this process.
Don't give the process, just return the final output in python dicts that only contains "expression" and "sentence".

######### Example ###########
input: 
{"intention": ["GET_WEATHER", "GET_LOCATION"], "slots": ["LOCATION", "LOCATION_MODIFIER"], "sentence": "i am interested in knowing the current weather conditions in the local area."}

output:
{"expression": "[IN:GET_WEATHER i am interested in knowing the current weather conditions [SL:LOCATION in the [IN:GET_LACATION [SL:LOCATION_MODIFIER local]] area].]",
 "sentence": "i am interested in knowing the current weather conditions in the local area."}

input:
{"intention": ["GET_SUNRISE", "GET_LOCATION"], "slots": ["LOCATION", "LOCATION_USER", "DATE_TIME"], "sentence": "what time does the sun rise in my hometown this saturday?"}

output:
{"expression": "[IN:GET_SUNRISE what time does the sun rise [SL:LOCATION in [IN:GET_LACATION [SL:LOCATION_USER my hometown]]] [SL:DATE_TIME this saturday]?]",
 "sentence": "what time does the sun rise in my hometown this saturday?"}

'''

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

def get_answer(query,choice):
    if choice == 1:
        content = query
    else:
        content = "input:\n" + "{"+query+"}\n\n" + "output:\n"
    r = openai.ChatCompletion.create(
        engine = model,
        temperature = 0.8,
        api_version = api_version_num,
        messages = [
            {
                "role":"system",
                "content":instruct1,
            },
            {
                "role":"user",
                "content":content,
            },
        ],
    )

    answer = r.choices[0]["message"]["content"]
    print(answer)
    return answer



def generate_from_gpt(datas,choice):
    results = []
    with ThreadPoolExecutor(max_workers=1000) as executor:
        responses = []
        for text in datas:
            res = executor.submit(get_answer,str(text),choice)
            responses.append(res)

    for resp in as_completed(responses):
        r = resp.result()
        results.append(r)
        print(r)
    return results


