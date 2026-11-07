"""A tiny support chat. Each chat has a session id; every call resends the history."""

import os

from openai import OpenAI

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
HISTORY: dict = {}


def handle_message(session_id: str, text: str) -> str:
    history = HISTORY.setdefault(session_id, [{"role": "system", "content": "You are a support agent."}])
    history.append({"role": "user", "content": text})
    reply = client.chat.completions.create(model="gpt-test", messages=history)
    answer = reply.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    return answer


if __name__ == "__main__":
    for message in ("Hi, my order is late", "It was order 1234"):
        print(handle_message("chat-0001", message))
