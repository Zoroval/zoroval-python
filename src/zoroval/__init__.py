"""Zoroval Python SDK: traces of your LLM app, grouped as whole conversations.

    import zoroval
    zoroval.init()                      # reads ZOROVAL_INGEST_KEY; wraps OpenAI and Anthropic

    with zoroval.conversation(chat_id, user_id=user.id):
        with zoroval.turn(input=user_text) as turn:
            reply = client.chat.completions.create(model=..., messages=history)
            turn.set_output(reply.choices[0].message.content)

The SDK never raises into your code, never blocks your requests, caps its memory, and exits
within `shutdown_timeout` seconds.
"""

from ._context import (Observation, conversation, current_observation, set_metadata, set_tags, set_user,
                       step, tool, trace, turn)
from ._setup import flush, init, instrument_anthropic, instrument_openai, shutdown, status
from ._version import __version__

__all__ = [
    "init", "shutdown", "flush", "status",
    "conversation", "turn", "trace", "step", "tool",
    "set_user", "set_tags", "set_metadata", "current_observation", "Observation",
    "instrument_openai", "instrument_anthropic",
    "__version__",
]
