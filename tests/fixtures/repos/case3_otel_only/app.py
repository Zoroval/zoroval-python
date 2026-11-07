"""Chat handler traced with plain OpenTelemetry (no LLM instrumentation yet)."""

import os

from openai import OpenAI

from tracing import provider, tracer

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))


def handle_message(conversation_id: str, text: str) -> str:
    with tracer.start_as_current_span("handle_message"):
        with tracer.start_as_current_span("load_profile"):
            pass
        reply = client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": text}])
        return reply.choices[0].message.content


if __name__ == "__main__":
    for message in ("Can I change my plan?", "To the yearly one"):
        print(handle_message("conv-0003", message))
    provider.force_flush()
