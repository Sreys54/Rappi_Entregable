from typing import Generator

from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are DataBot Rappi, an expert data analyst specializing in store operations.
You have access to historical data for the metric synthetic_monitoring_visible_stores from Rappi,
which counts how many stores are online and visible to users on the platform at each moment.

Guidelines:
- Answer concisely and use specific figures from the data summary provided
- Format large numbers with thousand separators (e.g., 5,432,100)
- Proactively highlight interesting operational patterns when relevant
- If a question cannot be answered from the available data, say so clearly
- Always respond in the same language the user writes in (Spanish or English)
"""

MODEL = "gemini-2.5-flash"


def init_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def stream_response(
    client: genai.Client,
    messages: list[dict],
    data_summary: str,
) -> Generator[str, None, None]:
    """
    Streams a Gemini response given the chat history and data context.

    The data summary is seeded as the first exchange in the API history so the
    model always has full context without re-sending it on every turn.

    Args:
        client: Initialized Gemini Client.
        messages: List of {"role": "user" | "assistant", "content": str}.
        data_summary: Pre-computed text summary of the dataset.

    Yields:
        Text chunks as they arrive from the Gemini streaming API.
    """
    # Seed the data context as the very first exchange so it is always
    # available to the model across the entire conversation.
    history = [
        types.Content(
            role="user",
            parts=[types.Part(text=f"Here is the dataset I will ask about:\n\n{data_summary}")],
        ),
        types.Content(
            role="model",
            parts=[types.Part(
                text="Understood. I have reviewed the Rappi store availability data summary. "
                     "What would you like to know?"
            )],
        ),
    ]

    # Append all previous turns except the current user message
    for msg in messages[:-1]:
        history.append(
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part(text=msg["content"])],
            )
        )

    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        history=history,
    )

    for chunk in chat.send_message_stream(messages[-1]["content"]):
        if chunk.text:
            yield chunk.text