"""Local PII masking implementation (PyDeid with regex fallback)."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .interface import PIIMaskerInterface


class LocalPIIMasker(PIIMaskerInterface):
    """PII masker that runs locally using PyDeid (if available)."""

    def __init__(self) -> None:
        self._deidentifier = None
        try:
            from pydeid import Deidentifier  # type: ignore

            self._deidentifier = Deidentifier()
        except Exception:
            self._deidentifier = None

    def mask_pii(self, text: str) -> Tuple[str, Dict]:
        if not text:
            return text, {}

        if self._deidentifier is not None:
            try:
                result = self._deidentifier.deidentify(text)
                masked_text = self._extract_text(result, default=text)
                entity_map = self._extract_entities(result)
                return masked_text, entity_map
            except Exception:
                return self._mask_with_regex(text)

        return self._mask_with_regex(text)

    @staticmethod
    def _extract_text(result: object, default: str) -> str:
        if isinstance(result, str):
            return result
        if hasattr(result, "text"):
            return getattr(result, "text")
        if isinstance(result, dict) and "text" in result:
            return str(result["text"])
        return default

    @staticmethod
    def _extract_entities(result: object) -> Dict:
        entity_map: Dict[str, Dict] = {}
        if hasattr(result, "entities"):
            entities = getattr(result, "entities", [])
            for entity in entities:
                text = getattr(entity, "text", "")
                label = getattr(entity, "label", "PII")
                replacement = getattr(entity, "replacement", f"[{label}]")
                entity_map[text] = {
                    "type": label,
                    "replacement": replacement,
                }
        return entity_map

    def _mask_with_regex(self, text: str) -> Tuple[str, Dict]:
        patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "PHONE": r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "DATE": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        }

        entity_map: Dict[str, Dict] = {}
        masked_text = text
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, masked_text):
                original = match.group(0)
                replacement = f"[{label}]"
                entity_map[original] = {"type": label, "replacement": replacement}
                masked_text = masked_text.replace(original, replacement)
        return masked_text, entity_map
