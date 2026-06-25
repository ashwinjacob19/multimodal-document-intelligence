import asyncio
import logging
import time

from sqlalchemy import select

from app.db.models.chunk import Chunk
from app.db.session import AsyncSessionLocal
from app.embeddings.factory import EmbeddingProviderFactory
from app.embeddings.provider import EmbeddingProvider

# Suppress verbose SQLAlchemy logging for clean output
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)


class SQLSuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return False


logging.getLogger("sqlalchemy.engine").addFilter(SQLSuppressFilter())
logging.getLogger("sqlalchemy.engine.Engine").addFilter(SQLSuppressFilter())
logging.getLogger("sqlalchemy.pool").addFilter(SQLSuppressFilter())


def run_verification_1(provider: EmbeddingProvider) -> None:
    """Verification 1: Embed 'Hello World' and print metadata and first five values."""
    print("--- Verification 1 ---")
    text = "Hello World"
    results = provider.embed([text])
    if not results:
        print("Error: No embedding generated.")
        return
    res = results[0]
    metadata = provider.metadata
    print(f"Model name: {metadata.model_name}")
    print(f"Embedding dimension: {metadata.dimension}")
    print(f"First five values: {res.embedding[:5]}")
    print()


async def run_verification_2(provider: EmbeddingProvider) -> None:
    """Verification 2: Embed all existing database chunks and print metrics."""
    print("--- Verification 2 ---")
    async with AsyncSessionLocal() as db:
        stmt = select(Chunk.text)
        res = await db.execute(stmt)
        chunk_texts = res.scalars().all()

    if not chunk_texts:
        print("No chunk records found in the database. Skipping Verification 2.")
        return

    # Generate embeddings and measure elapsed time
    start_time = time.time()
    provider.embed(list(chunk_texts))
    end_time = time.time()

    total_time = end_time - start_time
    metadata = provider.metadata
    print(f"Chunks processed: {len(chunk_texts)}")
    print(f"Model name: {metadata.model_name}")
    print(f"Embedding dimension: {metadata.dimension}")
    print(f"Total processing time: {total_time:.4f} seconds")
    print()


async def main() -> None:
    # Composition Root: instantiate provider once
    provider = EmbeddingProviderFactory.create()

    # Pass the provider down as a dependency
    run_verification_1(provider)
    await run_verification_2(provider)


if __name__ == "__main__":
    asyncio.run(main())
