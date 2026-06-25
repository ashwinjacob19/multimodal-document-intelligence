from app.config import settings
from app.embeddings.factory import EmbeddingProviderFactory
from app.embeddings.models import EmbeddingMetadata, EmbeddingResult
from app.embeddings.provider import EmbeddingProvider


def test_sentence_transformer_provider_metadata() -> None:
    """Verifies that provider metadata contains the correct model name

    and a valid, non-zero dimension.
    """
    provider = EmbeddingProviderFactory.create()
    metadata = provider.metadata

    assert metadata.model_name == settings.EMBEDDING_MODEL
    assert isinstance(metadata.dimension, int)
    assert metadata.dimension > 0


def test_embed_single_text() -> None:
    """Verifies embedding generation for a single text, checking its type,

    list format, and dynamic dimension length.
    """
    provider = EmbeddingProviderFactory.create()
    text = "Hello World"
    results = provider.embed([text])

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, EmbeddingResult)
    assert res.text == text
    assert isinstance(res.embedding, list)
    assert all(isinstance(val, float) for val in res.embedding)
    assert len(res.embedding) == provider.metadata.dimension


def test_embed_multiple_texts() -> None:
    """Verifies embedding generation for a batch of texts, confirming type,

    consistent dimensions, list structure, and float data type.
    """
    provider = EmbeddingProviderFactory.create()
    texts = ["Hello", "World", "Milestone 8"]
    results = provider.embed(texts)

    assert len(results) == len(texts)
    dim = provider.metadata.dimension
    for i, res in enumerate(results):
        assert isinstance(res, EmbeddingResult)
        assert res.text == texts[i]
        assert isinstance(res.embedding, list)
        assert len(res.embedding) == dim
        assert all(isinstance(val, float) for val in res.embedding)


def test_embed_empty_input() -> None:
    """Verifies that an empty input list returns an empty list result

    without raising any exceptions.
    """
    provider = EmbeddingProviderFactory.create()
    results = provider.embed([])
    assert results == []


def test_factory_singleton() -> None:
    """Verifies that the factory returns a cached singleton instance of the provider."""
    provider1 = EmbeddingProviderFactory.create()
    provider2 = EmbeddingProviderFactory.create()
    assert provider1 is provider2


# Fake implementation of EmbeddingProvider for dependency injection testing
class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata(model_name="fake-model", dimension=4)

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        return [
            EmbeddingResult(text=t, embedding=[0.1, 0.2, 0.3, 0.4])
            for t in texts
        ]


class TextPipeline:
    """A small test class demonstrating and verifying dependency injection."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def run_pipeline(self, texts: list[str]) -> list[float]:
        results = self.provider.embed(texts)
        return [sum(res.embedding) for res in results]


def test_dependency_injection_with_fake() -> None:
    """Verifies that consumer classes can receive an injected EmbeddingProvider

    interface and operate properly using mock/fake implementations.
    """
    fake_provider = FakeEmbeddingProvider()
    pipeline = TextPipeline(fake_provider)

    # Verify pipeline functions properly with the fake provider without factory calls
    sums = pipeline.run_pipeline(["hello", "world"])
    assert sums == [1.0, 1.0]
    assert pipeline.provider.metadata.model_name == "fake-model"
    assert pipeline.provider.metadata.dimension == 4
