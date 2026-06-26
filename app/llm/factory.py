import logging

from app.config import settings
from app.llm.provider import FakeLLMProvider, GeminiProvider, LLMProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    _instance: LLMProvider | None = None
    _logged_config: bool = False

    @classmethod
    def create(cls) -> LLMProvider:
        """Returns a singleton instance of the configured LLMProvider.

        Falls back to FakeLLMProvider if GEMINI_API_KEY is not configured.
        """
        if cls._instance is None:
            if not settings.GEMINI_API_KEY:
                cls._instance = FakeLLMProvider(
                    default_response=(
                        "Grounded mock answer (GEMINI_API_KEY not set)."
                    )
                )
                if not cls._logged_config:
                    logger.warning("LLM Provider: FakeLLMProvider")
                    logger.warning(
                        "GEMINI_API_KEY not configured. Using FakeLLMProvider."
                    )
                    cls._logged_config = True
            else:
                cls._instance = GeminiProvider(
                    api_key=settings.GEMINI_API_KEY,
                    model_name=settings.GEMINI_MODEL,
                )
                if not cls._logged_config:
                    logger.info(
                        f"LLM Provider: GeminiProvider\n"
                        f"Model: {settings.GEMINI_MODEL}"
                    )
                    cls._logged_config = True
        return cls._instance
