import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export async function POST(req: Request) {
  const { chatId, messages } = await req.json();
  const result = streamText({
    model: openai("gpt-test"),
    messages,
  });
  return result.toTextStreamResponse();
}
