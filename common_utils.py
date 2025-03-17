import io
import base64

from openai import OpenAI
from config import OPENAI_API_KEY

def encode_image_to_base64(pil_image):
    """
    Encodes a PIL image to base64 (PNG by default).
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def call_gpt_4(user_prompt, system_prompt: str = ""):
    """
    Calls GPT-4 (or GPT-4-like) with a list of message dicts.

    Example usage:
        user_prompt = [
                        {"type": "text", "text": "Generate 10 question and answer pairs based on the content of this page."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
                    ]
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your actual GPT-4 model name if needed
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content