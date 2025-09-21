from abc import ABC, abstractmethod
import config
from groq import Groq
from openai import OpenAI
import anthropic

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_answer(self, question: str) -> str:
        """
        Gets an answer for a given question from the LLM.

        Args:
            question: The question to ask the LLM.

        Returns:
            The answer from the LLM as a string.
        """
        pass

class GroqProvider(LLMProvider):
    """Provider for Groq API."""
    def __init__(self, model: str):
        self.model = model
        try:
            self.client = Groq(api_key=config.GROQ_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {e}")

    def get_answer(self, question: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API."""
    def __init__(self, model: str):
        self.model = model
        try:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def get_answer(self, question: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

class ClaudeProvider(LLMProvider):
    """Provider for Anthropic (Claude) API."""
    def __init__(self, model: str):
        self.model = model
        try:
            self.client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
            
    def get_answer(self, question: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        return message.content[0].text

