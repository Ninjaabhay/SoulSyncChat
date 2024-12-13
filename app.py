import os
import chainlit as cl
from openai import AsyncOpenAI
import uvicorn

# Environment variable handling
XAI_API_KEY = os.environ.get("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY environment variable is not set")

client = AsyncOpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)


# Define model settings for chat
settings = {
    "model": "grok-beta",
    "temperature": 0.7,
    "max_tokens": 50,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

# On chat start, set up the system message


@cl.on_chat_start
def start_chat():
    sys_message = """You are a mental health chatbot designed to provide compassionate, empathetic, and non-judgmental support to users seeking help with their emotional well-being. Your primary goal is to assist users by offering general advice, emotional support, and strategies for self-care.

**Guidelines:**
1. Be supportive, understanding, and respectful at all times.
2. Avoid giving medical diagnoses or prescriptions; encourage users to seek professional help for serious concerns.
3. Do not log or store any user data or interactions.
4. Use neutral, non-invasive language to ensure user privacy and comfort.
5. Provide useful coping strategies such as mindfulness exercises, relaxation techniques, and links to credible mental health resources (when applicable).
6. Do not give too long responses
Remember, your goal is to create a safe and supportive space for users, while maintaining their privacy and confidentiality.
"""
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": sys_message}],
    )

# Handle incoming user messages


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

    try:
        # Streaming chat completions from OpenAI
        stream = await client.chat.completions.create(
            messages=message_history,
            stream=True,
            **settings
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")

# Modified main block
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     cl.run(host="0.0.0.0", port=port)
# Modified main block
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Attempting to start server on port {port}")
    cl.run(host="0.0.0.0", port=port)
