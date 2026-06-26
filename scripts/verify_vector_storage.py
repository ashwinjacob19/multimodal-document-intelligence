import asyncio
import logging
import math

from sqlalchemy import text
from sqlalchemy.future import select

from app.db.models.chunk import Chunk
from app.db.session import AsyncSessionLocal

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
    print("--- Vector Storage Verification ---")
    async with AsyncSessionLocal() as db:
        # 1. Verify pgvector extension exists
        res = await db.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        ext_exists = res.scalar() is not None
        print(f"pgvector extension registered: {ext_exists}")

        # 2. Verify embedding column exists in chunks table
        res = await db.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'chunks' AND column_name = 'embedding'"
            )
        )
        col_exists = res.scalar() is not None
        print(f"chunks.embedding column exists: {col_exists}")

        # 3. Verify HNSW index exists on chunks table
        res = await db.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'chunks' AND indexname = 'idx_chunks_embedding'"
            )
        )
        indexdef = res.scalar()
        index_exists = indexdef is not None
        is_hnsw = "using hnsw" in indexdef.lower() if index_exists else False
        print(f"HNSW index 'idx_chunks_embedding' exists: {index_exists}")
        if index_exists:
            print(f"Index definition: {indexdef.strip()}")
            print(f"Index uses HNSW: {is_hnsw}")

        # 4. Fetch stored vector dimensions using vector_dims(embedding)
        res = await db.execute(
            text(
                "SELECT vector_dims(embedding) FROM chunks "
                "WHERE embedding IS NOT NULL"
            )
        )
        dims_sql = res.scalars().all()
        print(f"Chunks with embeddings: {len(dims_sql)}")

        if not dims_sql:
            print(
                "No stored vectors found in database chunks. "
                "Please upload a PDF to populate database first."
            )
            return

        # 5. Verify stored vector dimensions
        min_dim_sql = min(dims_sql)
        max_dim_sql = max(dims_sql)
        print(f"SQL vector_dims (min): {min_dim_sql}")
        print(f"SQL vector_dims (max): {max_dim_sql}")

        # 6. Fetch actual embeddings to compute dimensions & norms in Python
        stmt = select(Chunk.embedding).where(Chunk.embedding.isnot(None))
        res = await db.execute(stmt)
        embeddings = res.scalars().all()

        dims = [len(emb) for emb in embeddings]
        norms = [math.sqrt(sum(x * x for x in emb)) for emb in embeddings]

        min_dim = min(dims)
        max_dim = max(dims)
        avg_norm = sum(norms) / len(norms)

        print(f"Python calculated dimension (min): {min_dim}")
        print(f"Python calculated dimension (max): {max_dim}")
        print(f"Average vector norm (unit length check): {avg_norm:.6f}")



if __name__ == "__main__":
    asyncio.run(main())
