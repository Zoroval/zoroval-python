"""AWS Lambda chat handler. The container may freeze or stop right after each return."""

import json
import os

from openai import OpenAI

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))


def handler(event, context):
    body = json.loads(event["body"])
    reply = client.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": body["text"]}])
    return {"statusCode": 200, "body": json.dumps({"answer": reply.choices[0].message.content})}
