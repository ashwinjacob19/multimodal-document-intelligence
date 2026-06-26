import asyncio
import logging

from sqlalchemy import select

from app.db.models.document import Document, DocumentStatus
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.search.models import SearchQuery
from app.services.retrieval_service import RetrievalService
from app.services.search_service import SearchService

# Suppress verbose SQLAlchemy logging for clean output
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)


class SQLSuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return False


logging.getLogger("sqlalchemy.engine").addFilter(SQLSuppressFilter())
logging.getLogger("sqlalchemy.engine.Engine").addFilter(SQLSuppressFilter())
logging.getLogger("sqlalchemy.pool").addFilter(SQLSuppressFilter())


async def main() -> None:
    print("--- Retrieval Service Verification ---")
    provider = EmbeddingProviderFactory.create()
    search_service = SearchService(provider)
    retrieval_service = RetrievalService(search_service)

    async with AsyncSessionLocal() as db:
        # Check if we have at least one successfully processed document
        stmt = select(Document).where(Document.status == DocumentStatus.processed)
        res = await db.execute(stmt)
        docs = res.scalars().all()

        if not docs:
            print(
                "No processed documents found in the database. "
                "Please upload a PDF first."
            )
            return

        # Let's perform a query
        q_text = "page one text content"
        print(f"Executing retrieval query: '{q_text}'")
        query = SearchQuery(text=q_text, limit=3)
        context = await retrieval_service.retrieve(db, query)

        # Print structured retrieval information
        print(f"\nQuery          : '{context.metadata.query}'")
        print(f"Retrieved Count: {context.metadata.retrieved_count}")
        print(f"Document ID    : {context.metadata.document_id}")
        print(f"Limit          : {context.metadata.limit}")

        print("\nRetrieved Chunks:")
        for rank, chunk in enumerate(context.chunks, 1):
            text_preview = chunk.text.replace("\n", " ")
            if len(text_preview) > 50:
                text_preview = text_preview[:50] + "..."
            print(f"  Rank {rank}:")
            print(f"    Similarity : {chunk.similarity:.6f}")
            print(f"    Page Number: {chunk.page_number}")
            print(f"    Chunk Index: {chunk.chunk_index}")
            print(f"    Document ID: {chunk.document_id}")
            print(f"    Text       : '{text_preview}'")

        # Verify ordering is preserved
        search_results = await search_service.search(db, query)
        assert len(context.chunks) == len(search_results), "Mismatch in count"
        for ctx_c, s_c in zip(context.chunks, search_results, strict=True):
            assert ctx_c.chunk_id == s_c.chunk_id, "Rank ordering mismatch"
        print(
            "\nVerification completed successfully. "
            "Rank ordering is verified to be preserved."
        )


if __name__ == "__main__":
    asyncio.run(main())
