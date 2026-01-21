"""Tests for database models."""

from __future__ import annotations

from db.models import ChunkMetadata


def test_chunk_metadata_roundtrip() -> None:
    meta = ChunkMetadata(patient_id="p1", resource_id="r1", resource_type="Observation")
    data = meta.to_dict()
    loaded = ChunkMetadata.from_dict(data)
    assert loaded.patient_id == "p1"
    assert loaded.resource_id == "r1"
