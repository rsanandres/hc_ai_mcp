"""Tests for PII masking."""

from __future__ import annotations

from agent.pii_masker.factory import create_pii_masker
from agent.pii_masker.local_masker import LocalPIIMasker
from agent.pii_masker.interface import PIIMaskerInterface


def test_factory_returns_local_by_default() -> None:
    masker = create_pii_masker()
    assert isinstance(masker, LocalPIIMasker)
    assert isinstance(masker, PIIMaskerInterface)


def test_mask_email() -> None:
    masker = LocalPIIMasker()
    text = "Contact john.doe@example.com for details"
    masked, entities = masker.mask_pii(text)
    assert "john.doe@example.com" not in masked
    assert "[EMAIL]" in masked
    assert "john.doe@example.com" in entities


def test_mask_phone() -> None:
    masker = LocalPIIMasker()
    text = "Call me at 555-123-4567"
    masked, entities = masker.mask_pii(text)
    assert "555-123-4567" not in masked
    assert "[PHONE]" in masked


def test_mask_ssn() -> None:
    masker = LocalPIIMasker()
    text = "SSN is 123-45-6789"
    masked, entities = masker.mask_pii(text)
    assert "123-45-6789" not in masked
    assert "[SSN]" in masked


def test_mask_date() -> None:
    masker = LocalPIIMasker()
    text = "Date of birth: 01/15/1990"
    masked, entities = masker.mask_pii(text)
    assert "01/15/1990" not in masked
    assert "[DATE]" in masked


def test_mask_empty_string() -> None:
    masker = LocalPIIMasker()
    masked, entities = masker.mask_pii("")
    assert masked == ""
    assert entities == {}


def test_mask_no_pii() -> None:
    masker = LocalPIIMasker()
    text = "Patient has type 2 diabetes mellitus"
    masked, entities = masker.mask_pii(text)
    assert masked == text
    assert entities == {}


def test_mask_multiple_pii() -> None:
    masker = LocalPIIMasker()
    text = "Email: a@b.com, Phone: 555-111-2222, SSN: 999-88-7777"
    masked, entities = masker.mask_pii(text)
    assert "a@b.com" not in masked
    assert "555-111-2222" not in masked
    assert "999-88-7777" not in masked
    assert len(entities) == 3
