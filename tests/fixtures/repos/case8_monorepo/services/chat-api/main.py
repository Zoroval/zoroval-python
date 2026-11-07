from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI()


@app.post("/chat")
def chat(conversation_id: str, text: str) -> str:
    reply = client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": text}])
    return reply.choices[0].message.content
