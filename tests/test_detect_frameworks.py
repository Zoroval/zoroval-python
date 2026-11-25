"""zoroval detect: framework packages are case 1 (zoroval.init() records their OpenAI and Anthropic calls),
OpenLLMetry instrumentors for more providers are case 4, and a Gemini app gets init() plus a named
instrumentor (found 2026-11-25 by the SDK simulations)."""

import pytest

from zoroval.cli.detect import scan

APP = "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n"


def _scan(tmp_path, requirements, source=APP):
    (tmp_path / "requirements.txt").write_text(requirements)
    (tmp_path / "app.py").write_text(source)
    return scan(tmp_path)


@pytest.mark.parametrize("requirements", ["openai-agents\n", "langchain\nlangchain-anthropic\n",
                                          "langgraph\nlangchain-openai\n", "llama-index-core\nllama-index-llms-openai\n"])
def test_frameworks_are_case_1(tmp_path, requirements):
    report = _scan(tmp_path, requirements)
    assert 1 in report["cases"]
    assert any("zoroval.init()" in step for step in report["plan"])
    assert report["tracing"]["frameworks"]
    assert any("Do not add an OpenInference" in note for note in report["notes"])


def test_framework_import_without_manifest_entry(tmp_path):
    report = _scan(tmp_path, "", "from langchain_anthropic import ChatAnthropic\n" + APP)
    assert report["tracing"]["anthropic"] is True and 1 in report["cases"]


@pytest.mark.parametrize("package", ["opentelemetry-instrumentation-mistralai", "opentelemetry-instrumentation-groq",
                                     "opentelemetry-instrumentation-ollama", "opentelemetry-instrumentation-cohere"])
def test_more_openllmetry_instrumentors_are_case_4(tmp_path, package):
    report = _scan(tmp_path, f"{package}\n")
    assert 4 in report["cases"] and package in report["tracing"]["openllmetry"]


def test_contrib_v2_is_not_openllmetry(tmp_path):
    report = _scan(tmp_path, "opentelemetry-instrumentation-openai-v2\nopentelemetry-sdk\n")
    assert report["tracing"].get("openllmetry", []) == []


def test_gemini_plan_has_init_and_instrumentor(tmp_path):
    report = _scan(tmp_path, "google-genai>=3\n", "from google import genai\n" + APP)
    assert 1 in report["cases"]
    plan = " ".join(report["plan"])
    assert "zoroval.init()" in plan and "GoogleGenAIInstrumentor().instrument()" in plan
    assert "openinference-instrumentation-google-genai" in plan


def test_gemini_already_instrumented(tmp_path):
    report = _scan(tmp_path, "google-genai\nopeninference-instrumentation-google-genai\nopentelemetry-sdk\n")
    assert not any("GoogleGenAIInstrumentor" in step for step in report["plan"])
    assert any("zoroval.init()" in step for step in report["plan"])


def test_pydantic_ai_counts_only_on_openai_or_anthropic(tmp_path):
    source = "from pydantic_ai import Agent\nagent = Agent('openai:gpt-4o-mini')\n" + APP
    report = _scan(tmp_path, "pydantic-ai-slim[openai]\n", source)
    assert 1 in report["cases"] and report["tracing"]["openai"] is True


def test_pydantic_ai_on_another_provider_is_not_promised(tmp_path):
    source = "from pydantic_ai import Agent\nagent = Agent('google-gla:gemini-2.5-flash')\n" + APP
    report = _scan(tmp_path, "pydantic-ai-slim[google]\n", source)
    assert 1 not in report["cases"]
    assert any("Pydantic AI found" in note for note in report["notes"])
