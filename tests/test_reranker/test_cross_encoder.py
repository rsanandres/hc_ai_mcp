"""Tests for cross encoder reranker (without loading model)."""

from __future__ import annotations

from typing import Any, List

from reranker.cross_encoder import Reranker


class DummyModel:
    def predict(self, pairs: List[Any]) -> List[float]:
        return [float(i) for i, _ in enumerate(pairs)]


class DummyDoc:
    def __init__(self, text: str, doc_id: str) -> None:
        self.page_content = text
        self.metadata = {"chunkId": doc_id}


def _make_reranker() -> Reranker:
    reranker = Reranker.__new__(Reranker)
    reranker._model = DummyModel()
    reranker._device = "cpu"
    reranker._model_name = "dummy"
    return reranker


def test_rerank_with_scores_ordering() -> None:
    reranker = _make_reranker()
    docs = [DummyDoc("a", "1"), DummyDoc("b", "2"), DummyDoc("c", "3")]
    results = reranker.rerank_with_scores("q", docs)
    assert len(results) == 3
    assert results[0][0].page_content == "c"
