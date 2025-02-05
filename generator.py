
import os
from abc import ABC, abstractmethod

from openai import OpenAI
import google.generativeai as genai
import anthropic

###############################
# 3. GENERATOR CLASS (STUB)
###############################

class BaseGenerator(ABC):
    """Abstract generator interface."""

    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        pass

    # New method for structured blocks
    def generate_blocks(self, blocks: list, system_prompt: str = "") -> str:
        """
        Default implementation: flatten blocks into text
        (for text-only LLMs that can’t handle images).
        Subclasses that support images can override this.
        """
        # By default, just flatten everything into one string:
        text_parts = []
        for block in blocks:
            if block["type"] == "text":
                text_parts.append(block["text"])
            else:
                # For text-only LLMs, just mention that there's an image
                text_parts.append(f"[Image omitted: {block.get('mime_type','?')}]")

        combined = "\n".join(text_parts)
        return self.generate(combined, system_prompt=system_prompt)


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
    
    def generate_blocks(self, blocks: list, system_prompt: str = "") -> str:
        # Build messages = [{"role":"user","content":[ ...blocks... ]}]
        # where each block has "type":"text" or "type":"image_url"
        # e.g. {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
        # Then pass messages to openai_client.chat.completions.create(...)
        content_list = []
        for block in blocks:
            if block["type"] == "text":
                content_list.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                # GPT uses: {"type": "image_url","image_url": {"url": "data:image/png;base64,..."}}
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.get('mime_type','image/png')};base64,{block['data']}"
                    }
                })
            else:
                content_list.append({"type": "text", "text": f"[Unknown: {block}]"})
        
        user_message = {
            "role": "user",
            "content": content_list
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(user_message)

        response = self.openai_client.chat.completions.create(
            model="gpt-4v",
            messages=messages,
            max_tokens=300,
        )
        return response.choices[0].message.content
    
class GeminiGenerator(BaseGenerator):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """
        Fallback if someone calls generate() with just a string.
        We'll treat it as a single text prompt, no images.
        """
        if system_prompt:
            user_prompt = f"[SYSTEM PROMPT: {system_prompt}]\n\n{user_prompt}"
        response = self.model.generate_content([user_prompt])
        return response.text

    def generate_blocks(self, blocks: list, system_prompt: str = "") -> str:
        """
        Takes a list of blocks of form:
          [
            {"type":"text","text":"some user text"},
            {"type":"image","mime_type":"image/png","data":"<base64>"},
            ...
          ]
        Then passes them to Gemini as [image_dict, image_dict, ..., final_text].
        """
        # Convert your blocks into the Gemini input format
        gemini_payload = []
        for block in blocks:
            if block["type"] == "text":
                # We'll handle the system prompt by prepending it to the text
                # when we reach the *final* text block. But we might have multiple text blocks...
                # For simplicity, let's just store them in a list. We'll combine them at the end.
                gemini_payload.append(block["text"])
            elif block["type"] == "image":
                # Gemini expects {"mime_type":"image/png", "data":"<base64>"}
                gemini_payload.append({
                    "mime_type": block.get("mime_type", "image/png"),
                    "data": block["data"]
                })
            else:
                # Fallback for unknown block types
                gemini_payload.append(f"[Unsupported block type: {block['type']}]")

        # Combine all text blocks into a single string at the end.
        # Because Gemini typically does: [ {img}, {img}, ..., "prompt string" ]
        final_list = []
        combined_text = []
        for item in gemini_payload:
            # If it's a dict with "mime_type" and "data", it's an image
            if isinstance(item, dict) and "mime_type" in item and "data" in item:
                # first flush any text we have accumulated so far
                if combined_text:
                    text_prompt = "\n".join(combined_text)
                    if system_prompt:
                        text_prompt = f"[SYSTEM PROMPT: {system_prompt}]\n\n{text_prompt}"
                        system_prompt = ""  # only apply system prompt once
                    final_list.append(text_prompt)
                    combined_text.clear()
                final_list.append(item)  # the image
            else:
                # It's text
                combined_text.append(str(item))

        # If there's leftover text at the end, add it now
        if combined_text:
            text_prompt = "\n".join(combined_text)
            if system_prompt:
                text_prompt = f"[SYSTEM PROMPT: {system_prompt}]\n\n{text_prompt}"
            final_list.append(text_prompt)

        # Now final_list is something like:
        #   [ {img}, {img}, ..., "some final text" ]
        # Call Gemini
        response = self.model.generate_content(final_list)
        return response.text
