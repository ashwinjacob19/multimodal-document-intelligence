import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.llm.prompt_builder import PromptBuilder
from app.llm.provider import LLMProvider
from app.retrieval.models import RetrievalContext
from app.search.models import SearchQuery
from app.services.retrieval_service import RetrievalService


class AnswerService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
        prompt_builder: PromptBuilder,
    ):
        """Initializes AnswerService with injected dependencies."""
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder

    async def ask(
        self,
        db: AsyncSession,
        question: str,
        document_id: uuid.UUID | None = None,
        limit: int = settings.DEFAULT_SEARCH_LIMIT,
    ) -> tuple[str, RetrievalContext]:
        """Orchestrates RAG workflow to retrieve context and generate an answer."""
        # 1. Construct SearchQuery and perform retrieval
        query = SearchQuery(text=question, document_id=document_id, limit=limit)
        context = await self.retrieval_service.retrieve(db, query)

        # 2. Build structured prompt bundle using the prompt builder
        bundle = self.prompt_builder.build(question, context)

        # 3. Call the abstract LLMProvider to generate a grounded answer
        llm_response = await self.llm_provider.generate(
            bundle.system_prompt, bundle.user_prompt
        )

        return llm_response.text, context
