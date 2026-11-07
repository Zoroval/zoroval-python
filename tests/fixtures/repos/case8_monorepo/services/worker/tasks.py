import anthropic

client = anthropic.Anthropic()


def summarize(thread_id: str, text: str) -> str:
    message = client.messages.create(model="claude-test", max_tokens=200, messages=[{"role": "user", "content": text}])
    return message.content[0].text
