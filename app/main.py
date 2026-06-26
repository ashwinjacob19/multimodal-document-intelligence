from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.answer import router as answer_router
from app.api.documents import router as documents_router
from app.api.retrieval import router as retrieval_router
from app.api.search import router as search_router
from app.db.session import get_db
from app.llm.factory import LLMProviderFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to initialize and log LLM configuration."""
    LLMProviderFactory.create()
    yield


app = FastAPI(
    title="Enterprise Multimodal Document Intelligence Platform",
    description="Milestone 1 - Project Foundation",
    version="0.1.0",
    lifespan=lifespan,
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(retrieval_router)
app.include_router(answer_router)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """Basic health check to verify the application is running."""
    return {"status": "ok"}


@app.get("/ready", status_code=status.HTTP_200_OK)
async def ready(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Readiness check to verify external dependencies are reachable."""
    try:
        # Perform a simple query to check database connectivity
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {e}",
        ) from e
