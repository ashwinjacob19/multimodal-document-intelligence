import asyncio
from abc import ABC, abstractmethod

from app.llm.models import LLMResponse


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generates a text completion response for the system and user prompts."""
        pass


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str | None, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if not self._client:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generates content via the Gemini API using instructions and prompt."""
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )
        return LLMResponse(text=response.text)


class FakeLLMProvider(LLMProvider):
    def __init__(self, default_response: str = "Fake answer"):
        self.default_response = default_response
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Records prompts and returns the predefined mock response."""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return LLMResponse(text=self.default_response)
