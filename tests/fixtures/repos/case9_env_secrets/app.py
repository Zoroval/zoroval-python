"""Chat app that loads settings from .env in development and from Railway variables in production."""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))


def answer(chat_id: str, text: str) -> str:
    reply = client.chat.completions.create(model=os.environ.get("OPENAI_MODEL", "gpt-test"),
                                           messages=[{"role": "user", "content": text}])
    return reply.choices[0].message.content


if __name__ == "__main__":
    for message in ("Do you ship to Canada?", "How long does it take?"):
        print(answer("chat-0009", message))
