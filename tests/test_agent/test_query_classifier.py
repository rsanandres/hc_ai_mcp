"""Tests for query classifier."""

from __future__ import annotations

from agent.query_classifier import QueryClassifier, QueryType, classify_query


def test_greeting_classified_as_conversational() -> None:
    result = classify_query("Hello!")
    assert result.query_type == QueryType.CONVERSATIONAL


def test_medical_query_classified_as_medical() -> None:
    result = classify_query("What medications is the patient taking?")
    assert result.query_type == QueryType.MEDICAL


def test_mixed_query_with_greeting() -> None:
    result = classify_query("Hello, what are the patient's lab results?")
    assert result.query_type == QueryType.MIXED
    assert result.should_acknowledge_greeting is True


def test_icd10_code_detected_as_medical() -> None:
    result = classify_query("Look up ICD-10 code E11.9")
    assert result.query_type == QueryType.MEDICAL


def test_empty_query_is_unclear() -> None:
    result = classify_query("")
    assert result.query_type == QueryType.UNCLEAR


def test_who_are_you_is_conversational() -> None:
    result = classify_query("Who are you?")
    assert result.query_type == QueryType.CONVERSATIONAL


def test_calculate_gfr_is_medical() -> None:
    result = classify_query("Calculate the GFR for this patient")
    assert result.query_type == QueryType.MEDICAL


def test_thank_you_is_conversational() -> None:
    result = classify_query("Thank you!")
    assert result.query_type == QueryType.CONVERSATIONAL


def test_session_context_fallback() -> None:
    """When no keywords match, session context should influence classification."""
    classifier = QueryClassifier()
    result = classifier.classify(
        "tell me more",
        session_context={"last_query_type": "medical"},
        use_llm_fallback=False,
    )
    assert result.query_type == QueryType.MEDICAL
    assert result.method == "context"


def test_confidence_is_float() -> None:
    result = classify_query("Hello")
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_processing_time_tracked() -> None:
    result = classify_query("What is the blood pressure?")
    assert result.processing_time_ms >= 0.0
