import time

from openai import AsyncOpenAI, OpenAI, NOT_GIVEN


api_key = 'sk-CrQb5MFkrdZafRMnoNTfpNEyRiRJrzjIXOn3iF0BxQuB0F0M'
aclient_gpt = AsyncOpenAI(api_key=api_key,
                          base_url="https://api.chatanywhere.tech/v1")
client_gpt = OpenAI(api_key=api_key,
                    base_url="https://api.chatanywhere.tech/v1")


async def async_query_gpt(user_prompt: str, model="gpt-4o-mini-ca", temperature=NOT_GIVEN) -> str:
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


def query_gpt(user_prompt: str, model="gpt-4o-mini-ca", temperature=NOT_GIVEN) -> str:
    if isinstance(user_prompt, str):
        prompt = [{"role": "user", "content": user_prompt}]
    else:
        prompt = user_prompt

    try_call = 10
    while try_call:
        try_call -= 1
        try:
            response = client_gpt.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature
            )

            return response.choices[0].message.content
        except Exception as e:
            print("query_gpt error: ", e)
            time.sleep(20)
