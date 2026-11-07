"""Run in a fresh process: `langfuse.openai` patches the OpenAI client for good once imported.

Usage: python tests/coexist_langfuse.py <langfuse_first|zoroval_first>
Prints one JSON line with what each backend received.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from testing.fake_gateway import FakeGateway  # noqa: E402
from testing.fake_llm import FakeLLM  # noqa: E402
from testing.langfuse_standin import LangfuseStandIn  # noqa: E402

import zoroval  # noqa: E402

order = sys.argv[1]
gw, llm, lf = FakeGateway().start(), FakeLLM().start(), LangfuseStandIn().start()
key = gw.issue_key()
os.environ["OPENAI_API_KEY"] = "test-not-a-real-key"


def start_langfuse():
    from langfuse import Langfuse
    client = Langfuse(public_key="pk-lf-test-only", secret_key="sk-lf-test-only", base_url=lf.url,
                      flush_interval=0.1)
    from langfuse.openai import openai as lf_openai
    return client, lf_openai


if order == "langfuse_first":
    langfuse, openai_module = start_langfuse()
    zoroval.init(api_key=key, ingest_url=gw.url, flush_interval=0.05, service_name="coexist")
else:
    zoroval.init(api_key=key, ingest_url=gw.url, flush_interval=0.05, service_name="coexist")
    langfuse, openai_module = start_langfuse()

from langfuse import observe  # noqa: E402

client = openai_module.OpenAI(base_url=llm.openai_base_url, max_retries=0)


@observe()
def handle(question: str) -> str:
    reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": question}])
    return reply.choices[0].message.content


with zoroval.conversation("conv-lf"):
    answers = [handle("first"), handle("second")]

langfuse.flush()
zoroval.flush(5.0)


def is_generation(span):
    attrs = span["attributes"]
    return attrs.get("langfuse.observation.type") == "generation" or \
        attrs.get("zoroval.observation.type") == "generation"


zoroval_gens = [s for s in gw.state.spans if is_generation(s)]
langfuse_gens = [s for s in lf.spans if is_generation(s)]
print(json.dumps({
    "answers": answers,
    "zoroval_generations": len(zoroval_gens),
    "langfuse_generations": len(langfuse_gens),
    "zoroval_generation_conversations": sorted({s["attributes"].get("gen_ai.conversation.id") or
                                                s["attributes"].get("zoroval.conversation.id") or ""
                                                for s in zoroval_gens}),
    "zoroval_generation_has_output": all(any("output" in k for k in s["attributes"]) for s in zoroval_gens),
    "zoroval_traces": len(gw.grouped_traces()),
    "zoroval_status": {k: v for k, v in zoroval.status().items() if k in ("provider", "instrumented")},
}))
zoroval.shutdown()
langfuse.shutdown()
