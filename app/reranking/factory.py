import logging

from app.config import settings
from app.reranking.provider import (
    CrossEncoderRerankProvider,
    FakeRerankProvider,
    RerankProvider,
)

logger = logging.getLogger(__name__)


class RerankProviderFactory:
    _instance: RerankProvider | None = None
    _logged_config: bool = False

    @classmethod
    def create(cls) -> RerankProvider:
        """Returns a singleton instance of the configured RerankProvider."""
        if cls._instance is None:
            if settings.RERANK_PROVIDER == "sbert":
                cls._instance = CrossEncoderRerankProvider(
                    settings.RERANK_MODEL
                )
                if not cls._logged_config:
                    logger.info(
                        f"Rerank Provider: CrossEncoderRerankProvider\n"
                        f"Model: {settings.RERANK_MODEL}"
                    )
                    cls._logged_config = True
            else:
                cls._instance = FakeRerankProvider()
                if not cls._logged_config:
                    logger.warning("Rerank Provider: FakeRerankProvider")
                    cls._logged_config = True
        return cls._instance
