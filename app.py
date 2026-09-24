from openai import OpenAI
from pprint import pprint

client = OpenAI(
    base_url="http://192.168.1.168:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit-mlx",
    messages=[
        {
            "role": "system",
            "content": "You are an expert assistant for an industrial Ethernet switch."
        },
        {
            "role": "user",
            "content": "Show me how to disable Ethernet port 4."
        }
    ]
)

print(response.choices[0].message.content)
pprint(response)