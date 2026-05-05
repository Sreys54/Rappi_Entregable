from typing import Generator

# google-genai es el SDK oficial nuevo (>= 1.0.0).
# No confundir con el paquete deprecado "google-generativeai".
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

# gemini-2.5-flash: capa gratuita con cuota disponible.
# gemini-2.0-flash tenía limit=0 en la capa gratuita al momento del desarrollo.
MODEL = "gemini-2.5-flash"


def init_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def stream_response(
    client: genai.Client,
    messages: list[dict],
    data_summary: str,
) -> Generator[str, None, None]:
    # Sembramos el resumen como el primer intercambio del historial (user→model)
    # en lugar de inyectarlo en cada mensaje del usuario. Así el modelo siempre
    # tiene el contexto disponible sin reenviar ~3 KB en cada turno.
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

    # Agregamos todos los turnos previos EXCEPTO el último mensaje del usuario,
    # que se envía por separado en send_message_stream como el turno actual.
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

    # El generador cede fragmentos de texto a medida que llegan;
    # Streamlit los renderiza en tiempo real con st.write_stream().
    for chunk in chat.send_message_stream(messages[-1]["content"]):
        if chunk.text:
            yield chunk.text