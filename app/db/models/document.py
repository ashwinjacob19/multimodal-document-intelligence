import enum
import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class DocumentType(enum.StrEnum):
    pdf = "pdf"
    docx = "docx"
    image = "image"
    url = "url"


class DocumentStatus(enum.StrEnum):
    uploaded = "uploaded"
    processing = "processing"
    processed = "processed"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID, primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    document_type: Mapped[DocumentType] = mapped_column(
        sa.Enum(DocumentType, name="documenttype"), nullable=False
    )
    status: Mapped[DocumentStatus] = mapped_column(
        sa.Enum(DocumentStatus, name="documentstatus"),
        nullable=False,
        default=DocumentStatus.uploaded,
    )
    source: Mapped[str | None] = mapped_column(sa.String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
    )
