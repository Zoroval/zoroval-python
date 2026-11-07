"""Chat app on the Langfuse SDK v2 (not OpenTelemetry based)."""

from langfuse.decorators import observe
from langfuse.openai import openai


@observe()
def handle_message(session_id: str, text: str) -> str:
    reply = openai.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": text}],
                                           session_id=session_id)
    return reply.choices[0].message.content
