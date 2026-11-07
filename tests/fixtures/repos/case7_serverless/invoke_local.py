"""Two warm invocations of the same chat, then the container stops without running exit hooks."""

import json
import os

from handler import handler

for text in ("Is the store open?", "And on Sunday?"):
    event = {"body": json.dumps({"session_id": "chat-0007", "text": text})}
    print(handler(event, None)["body"], flush=True)
os._exit(0)  # like a frozen or killed Lambda container: no atexit
