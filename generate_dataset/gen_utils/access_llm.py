import time

from openai import AsyncOpenAI, OpenAI


model = "deepseek-chat"
api_key = 'sk-bd3ccb0c1ceb43e0b8210a90657b7bb4'
aclient_gpt = AsyncOpenAI(api_key=api_key,
                          base_url="https://api.deepseek.com/")
# client_gpt = OpenAI(api_key=api_key,
#                           base_url="https://api.deepseek.com/")
# deepseek-chat

# model = "gpt-4o-mini"
# api_key = 'sk-56a4ejBaNLcRbTiZfnbIX6L6i3lDzuCdscWRhPO1TR5tQfnJ'
# aclient_gpt = AsyncOpenAI(api_key=api_key,
#                           base_url="https://api.chatanywhere.tech/v1")
# client_gpt = OpenAI(api_key=api_key,
#                     base_url="https://api.chatanywhere.tech/v1")


#chatgpt


# client_gpt = OpenAI(api_key="sk-1234",
#                           base_url="https://models.kclab.cloud")
# aclient_gpt = AsyncOpenAI(api_key="sk-1234",
#                           base_url="https://models.kclab.cloud")

# model = "qwen2.5-72b-instruct"
# api_key = 'sk-24121fb6f64b43e6bbdbc54fce526c7a'
# aclient_gpt = AsyncOpenAI(api_key=api_key,
#                           base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# client_gpt = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key=api_key,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# client = OpenAI(api_key="6137bd4d-1a18-4159-ad53-8dded07d3e1d", base_url="https://ark.cn-beijing.volces.com/api/v3")
# model = "deepseek-v3-241226"

async def async_query_gpt(user_prompt: str, model=model, temperature=0.3) -> str:
    # response = await client.chat.completions.create(
    #             model=model,
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful assistant"},
    #                 {"role": "user", "content": user_prompt},
    #             ],
    #         )
    response = await aclient_gpt.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=temperature,
    )
    return response.choices[0].message.content


def query_gpt(user_prompt: str, model=model, temperature=0.3) -> str:
    if isinstance(user_prompt, str):
        prompt = [{"role": "user", "content": user_prompt}]
    else:
        prompt = user_prompt

    try_call = 10
    while try_call:
        try_call -= 1
        try:
            response = client.chat.completions.create(  # fixme: conic分支后有报错，后期处理下
                model=model,
                messages=prompt,
                temperature=temperature
            )

            return response.choices[0].message.content
        except Exception as e:
            print("query_gpt error: ", e)
            time.sleep(20)
