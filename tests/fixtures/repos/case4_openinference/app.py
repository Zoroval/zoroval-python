"""Chat app instrumented with OpenInference (as LangChain or LlamaIndex users often are)."""

import os

from openai import OpenAI
from openinference.instrumentation import using_session
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

provider = TracerProvider()
trace.set_tracer_provider(provider)
OpenAIInstrumentor().instrument(tracer_provider=provider)

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))


def handle_message(session_id: str, text: str) -> str:
    with using_session(session_id):
        reply = client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": text}])
    return reply.choices[0].message.content


if __name__ == "__main__":
    for message in ("Book a table for two", "Make it 8 pm"):
        print(handle_message("chat-0004", message))
