"""Factory for creating PII maskers."""

from __future__ import annotations

import os

from .aws_masker import AWSComprehendMedicalMasker
from .interface import PIIMaskerInterface
from .local_masker import LocalPIIMasker


def create_pii_masker() -> PIIMaskerInterface:
    provider = os.getenv("PII_MASKER_PROVIDER", "local").lower()
    if provider in {"aws", "comprehend"}:
        return AWSComprehendMedicalMasker()
    return LocalPIIMasker()
