from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class EmbeddingMetadata:
    model_name: str
    dimension: int
