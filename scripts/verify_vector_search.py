import asyncio
import logging

from sqlalchemy import select

from app.db.models.document import Document, DocumentStatus
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.search.models import SearchQuery
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
    print("--- Semantic Search Verification ---")
    provider = EmbeddingProviderFactory.create()
    search_service = SearchService(provider)

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

        print(f"Found {len(docs)} processed documents in the database.")
        for doc in docs:
            print(f"  - Document: {doc.filename} (ID: {doc.id})")

        # Let's perform some queries
        queries = ["page one text content", "page two text content"]
        for q in queries:
            print(f"\nExecuting semantic search query: '{q}'")
            query = SearchQuery(text=q, limit=3)
            results = await search_service.search(db, query)

            print(f"Results for '{q}':")
            if not results:
                print("  No matching chunks found.")
            for rank, res_item in enumerate(results, 1):
                text_preview = res_item.text.replace("\n", " ")
                if len(text_preview) > 50:
                    text_preview = text_preview[:50] + "..."
                print(f"  Rank {rank}:")
                print(f"    Similarity : {res_item.similarity:.6f}")
                print(f"    Document ID: {res_item.document_id}")
                print(f"    Chunk Index: {res_item.chunk_index}")
                print(f"    Page Number: {res_item.page_number}")
                print(f"    Text       : '{text_preview}'")

        print("\nVerification completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
