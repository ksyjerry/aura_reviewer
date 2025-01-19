from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "2022년 월드컵 우승팀은?"
        }
    ]
)

print(completion.choices[0].message.content)
