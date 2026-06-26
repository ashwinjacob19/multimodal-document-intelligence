import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document, DocumentStatus, DocumentType
from app.db.session import get_db
from app.embeddings.factory import EmbeddingProviderFactory
from app.embeddings.provider import EmbeddingProvider
from app.processing.processor import DocumentProcessor
from app.services.storage import StorageService

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/png",
    "image/jpeg",
    "image/jpg",
}

EXTENSION_TO_DOC_TYPE = {
    ".pdf": DocumentType.pdf,
    ".docx": DocumentType.docx,
    ".png": DocumentType.image,
    ".jpg": DocumentType.image,
    ".jpeg": DocumentType.image,
}

# Instantiate storage service
storage_service = StorageService()


# Dependency injection helpers
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProviderFactory.create()


def get_document_processor(
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentProcessor:
    return DocumentProcessor(provider)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    processor: DocumentProcessor = Depends(get_document_processor),
) -> dict:
    """Uploads a document, saves it to disk, and registers it in PostgreSQL."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File has no name.",
        )

    # Extract extension
    _, ext = os.path.splitext(file.filename.lower())

    # Validate both file extension and MIME type
    if ext not in ALLOWED_EXTENSIONS or file.content_type not in ALLOWED_MIME_TYPES:
        allowed_exts = sorted(ALLOWED_EXTENSIONS)
        allowed_mimes = sorted(ALLOWED_MIME_TYPES)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type. Supported extensions: {allowed_exts}. "
                f"Supported MIME types: {allowed_mimes}."
            ),
        )

    # Generate unique filename preserving the extension
    unique_filename = f"{uuid.uuid4()}{ext}"

    try:
        # Save file to disk using the storage service
        storage_path = storage_service.save_file(file, unique_filename)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        ) from e

    # Determine document type based on extension
    doc_type = EXTENSION_TO_DOC_TYPE[ext]

    # Create a corresponding Document record in PostgreSQL
    new_doc = Document(
        filename=file.filename,
        document_type=doc_type,
        status=DocumentStatus.uploaded,
        source=storage_path,
    )

    try:
        db.add(new_doc)
        await db.commit()
        await db.refresh(new_doc)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database transaction failed: {e}",
        ) from e

    # Invoke the document processor to start pipeline
    await processor.process(new_doc, db)

    return {
        "document_id": str(new_doc.id),
        "filename": new_doc.filename,
        "status": new_doc.status,
    }

