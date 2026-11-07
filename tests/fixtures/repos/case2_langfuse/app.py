"""Chat app already traced with Langfuse (SDK v3+, OpenTelemetry based)."""

import os

from langfuse import Langfuse, observe, propagate_attributes
from langfuse.openai import OpenAI

langfuse = Langfuse()
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))


@observe()
def handle_message(session_id: str, text: str) -> str:
    with propagate_attributes(session_id=session_id, user_id="user-2"):
        reply = client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": text}])
    return reply.choices[0].message.content


if __name__ == "__main__":
    for message in ("Where is my refund?", "It was last Tuesday"):
        print(handle_message("chat-0002", message))
    langfuse.flush()
