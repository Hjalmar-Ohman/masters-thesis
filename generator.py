
import os
from abc import ABC, abstractmethod

from openai import OpenAI
###############################
# 3. GENERATOR CLASS (STUB)
###############################

class BaseGenerator(ABC):
    """Abstract generator interface."""
    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        pass


class GPTGenerator(BaseGenerator):
    """
    A stub generator that simulates an LLM call.
    Replace the body of generate() with your actual API call to GPT/Claude/Gemini.
    """

    def __init__(self, api_key: str):
        self.openai_client = OpenAI(api_key=api_key)


    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
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

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with your actual GPT-4 model name if needed
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content