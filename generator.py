
from abc import ABC, abstractmethod


###############################
# 3. GENERATOR CLASS (STUB)
###############################

class BaseGenerator(ABC):
    """Abstract generator interface."""
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GPTGenerator(BaseGenerator):
    """
    A stub generator that simulates an LLM call.
    Replace the body of generate() with your actual API call to GPT/Claude/Gemini.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        # Replace with an actual API call.
        return f"[Simulated GPT Response]\nPrompt:\n{prompt}"