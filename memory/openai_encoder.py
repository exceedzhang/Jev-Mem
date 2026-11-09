"""Jev-Mem embeddings using the existing OpenAI/Azure v1 configuration."""

import os
import numpy as np
from openai import OpenAI

_AUTO = ("", "auto", "0", "none")


class OpenAIVectorEncoder:
    """OpenAI-compatible embedding adapter.

    ``dimensions`` is optional: when configured (constructor argument or
    ``OPENAI_EMBEDDING_DIMENSIONS``) it is sent as-is (OpenAI Matryoshka
    truncation). When unset/empty, the encoder probes the endpoint once and
    uses the model's native dimension without sending the parameter --
    vLLM-served models such as Qwen3-Embedding reject it with HTTP 400.
    """

    def __init__(self, model_name="text-embedding-3-small", client=None, dimension=None):
        self.model_name = os.getenv("OPENAI_EMBEDDING_MODEL", model_name)
        requested = dimension if dimension is not None else os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "")
        if isinstance(requested, str) and requested.strip().lower() in _AUTO:
            requested = None
        self.send_dimensions = requested is not None
        self.dimension = int(requested) if requested is not None else None
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.client = client if client is not None else OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        if self.dimension is None:
            self.dimension = self._probe_dimension()

    def _probe_dimension(self):
        """Ask the endpoint for one vector and adopt its native dimension."""
        response = self.client.embeddings.create(model=self.model_name, input=["dimension probe"])
        if not response.data or not response.data[0].embedding:
            raise ValueError("Embedding dimension probe returned no vector")
        return len(response.data[0].embedding)

    def _embed(self, client, batch):
        kwargs = {"model": self.model_name, "input": batch}
        if self.send_dimensions:
            kwargs["dimensions"] = self.dimension
        return client.embeddings.create(**kwargs)

    def encode(self, texts, timeout_seconds=None):
        return self.encode_batch([texts] if isinstance(texts, str) else texts, timeout_seconds=timeout_seconds)

    def encode_batch(self, texts, batch_size=100, timeout_seconds=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = []
        client = self.client
        if timeout_seconds is not None:
            client = client.with_options(timeout=max(0.001, timeout_seconds / 4), max_retries=0)
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            response = self._embed(client, batch)
            data = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in data] != list(range(len(batch))):
                raise ValueError("Embedding response does not match input batch")
            vectors.extend(item.embedding for item in data)
        result = np.asarray(vectors, dtype=np.float32)
        if result.shape != (len(texts), self.dimension) or not np.isfinite(result).all():
            raise ValueError("Unexpected embedding dimensions or non-finite values")
        return result
