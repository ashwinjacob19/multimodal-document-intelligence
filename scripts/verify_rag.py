import asyncio
import logging

from sqlalchemy import select

from app.db.models.document import Document, DocumentStatus
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.llm.factory import LLMProviderFactory
from app.llm.prompt_builder import GeminiPromptBuilder
from app.llm.provider import FakeLLMProvider, GeminiProvider
from app.services.answer_service import AnswerService
from app.services.retrieval_service import RetrievalService
from app.services.search_service import SearchService

# Suppress verbose SQLAlchemy logging
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)


class SQLSuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return False


logging.getLogger("sqlalchemy.engine").addFilter(SQLSuppressFilter())
logging.getLogger("sqlalchemy.engine.Engine").addFilter(SQLSuppressFilter())
logging.getLogger("sqlalchemy.pool").addFilter(SQLSuppressFilter())


async def main() -> None:
    print("--- Gemini RAG Integration Verification ---")

    # 1. Initialize services
    embed_provider = EmbeddingProviderFactory.create()
    search_service = SearchService(embed_provider)
    retrieval_service = RetrievalService(search_service)
    prompt_builder = GeminiPromptBuilder()
    llm_provider = LLMProviderFactory.create()

    # Report active provider configuration
    if isinstance(llm_provider, GeminiProvider):
        print("Provider: GeminiProvider")
        print(f"Model: {llm_provider.model_name}")
    elif isinstance(llm_provider, FakeLLMProvider):
        print("Provider: FakeLLMProvider")
        print("Reason: GEMINI_API_KEY not configured")
    else:
        print(f"Provider: {type(llm_provider).__name__}")

    answer_service = AnswerService(
        retrieval_service, llm_provider, prompt_builder
    )

    async with AsyncSessionLocal() as db:
        # Load a processed document from database
        stmt = select(Document).where(
            Document.status == DocumentStatus.processed
        )
        res = await db.execute(stmt)
        docs = res.scalars().all()

        if not docs:
            print("No processed documents found. Please upload a PDF first.")
            return

        doc = docs[0]
        print(f"\nLoaded document: {doc.filename} (ID: {doc.id})")

        question = "What is the content of page one?"
        print(f"Asking Question: '{question}'")

        # Execute RAG query (limit context to 2 chunks)
        try:
            answer, context = await answer_service.ask(
                db=db,
                question=question,
                document_id=doc.id,
                limit=2,
            )

            print("\nRetrieved Context Chunks:")
            for rank, chunk in enumerate(context.chunks, 1):
                text_preview = chunk.text.replace("\n", " ")
                if len(text_preview) > 60:
                    text_preview = text_preview[:60] + "..."
                print(f"  Rank {rank}:")
                print(f"    Page Number: {chunk.page_number}")
                print(f"    Chunk Index: {chunk.chunk_index}")
                print(f"    Similarity : {chunk.similarity:.6f}")
                print(f"    Text       : '{text_preview}'")

            print(f"\nGenerated Answer:\n{answer}\n")
            print("Verification completed successfully.")
        except Exception as e:
            print(f"\n[Error] Gemini RAG API call failed: {e}")
            print(
                "Verification completed with errors. "
                "API call was not successful."
            )


if __name__ == "__main__":
    asyncio.run(main())
