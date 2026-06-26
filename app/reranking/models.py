from dataclasses import dataclass


@dataclass(frozen=True)
class RerankMetadata:
    provider: str
    model_name: str
