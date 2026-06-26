import asyncio
import logging

from sqlalchemy import select

from app.db.models.document import Document, DocumentStatus
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.llm.factory import LLMProviderFactory
from app.llm.prompt_builder import GeminiPromptBuilder
from app.reranking.factory import RerankProviderFactory
from app.search.models import SearchQuery
from app.services.answer_service import AnswerService
from app.services.rerank_service import RerankService
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
    print("--- Retrieval Reranking Verification ---")

    # 1. Initialize services
    embed_provider = EmbeddingProviderFactory.create()
    search_service = SearchService(embed_provider)
    retrieval_service = RetrievalService(search_service)
    prompt_builder = GeminiPromptBuilder()

    # Resolve Reranking and LLM Providers
    rerank_provider = RerankProviderFactory.create()
    rerank_service = RerankService(rerank_provider)

    llm_provider = LLMProviderFactory.create()
    answer_service = AnswerService(
        retrieval_service, rerank_service, llm_provider, prompt_builder
    )

    print(f"Rerank Provider: {rerank_provider.metadata.provider}")
    print(f"Rerank Model   : {rerank_provider.metadata.model_name}")

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

        # 2. Retrieve initial semantic chunks
        query = SearchQuery(text=question, document_id=doc.id, limit=3)
        retrieval_context = await retrieval_service.retrieve(db, query)

        print("\n--- Original Semantic Retrieval Order ---")
        for rank, chunk in enumerate(retrieval_context.chunks, 1):
            text_preview = chunk.text.replace("\n", " ")
            if len(text_preview) > 50:
                text_preview = text_preview[:50] + "..."
            print(
                f"  Rank {rank}: [Index {chunk.chunk_index}] "
                f"Similarity: {chunk.similarity:.6f} "
                f"Text: '{text_preview}'"
            )

        # 3. Perform Reranking
        reranked_chunks = await rerank_service.rerank(
            question, retrieval_context.chunks
        )

        print("\n--- Reranked Context Order ---")
        for rank, chunk in enumerate(reranked_chunks, 1):
            text_preview = chunk.text.replace("\n", " ")
            if len(text_preview) > 50:
                text_preview = text_preview[:50] + "..."
            print(
                f"  Rank {rank}: [Index {chunk.chunk_index}] "
                f"Similarity: {chunk.similarity:.6f} "
                f"Rerank Score: {chunk.rerank_score:.6f} "
                f"Text: '{text_preview}'"
            )

        # 4. Verify AnswerService utilizes reranked context
        print("\nRunning AnswerService.ask()...")
        try:
            answer, answer_context = await answer_service.ask(
                db=db,
                question=question,
                document_id=doc.id,
                limit=3,
            )

            # Print first reranked chunk text
            first_chunk_text = answer_context.chunks[0].text.replace("\n", " ")
            if len(first_chunk_text) > 50:
                first_chunk_text = first_chunk_text[:50] + "..."

            print(f"Answer Context First Chunk: '{first_chunk_text}'")
            print(f"Answer Context Chunks Count: {len(answer_context.chunks)}")
            print(f"\nGenerated Answer:\n{answer}\n")

            # Verify AnswerService context order matches reranked_chunks order
            assert len(answer_context.chunks) == len(reranked_chunks)
            for a_chunk, r_chunk in zip(
                answer_context.chunks, reranked_chunks, strict=True
            ):
                assert a_chunk.chunk_id == r_chunk.chunk_id
                assert a_chunk.rerank_score == r_chunk.rerank_score

            print(
                "Verification completed successfully. "
                "AnswerService uses reranked chunks."
            )
        except Exception as e:
            print(f"\n[Error] AnswerService Q&A failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
