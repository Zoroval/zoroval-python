"""Chat app traced with LangSmith."""

from langsmith import traceable
from openai import OpenAI

client = OpenAI()


@traceable
def handle_message(thread_id: str, text: str) -> str:
    reply = client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": text}])
    return reply.choices[0].message.content
