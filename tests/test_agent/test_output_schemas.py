"""Tests for output schema parsing."""

from __future__ import annotations

from agent.output_schemas import (
    ValidatorOutput,
    ResearcherOutput,
    parse_validator_output,
)


def test_parse_pass_from_text() -> None:
    text = "VALIDATION_STATUS: PASS\nAll good."
    result = parse_validator_output(text)
    assert result.validation_status == "PASS"


def test_parse_fail_from_text() -> None:
    text = "VALIDATION_STATUS: FAIL\nBad data."
    result = parse_validator_output(text)
    assert result.validation_status == "FAIL"


def test_parse_needs_revision_default() -> None:
    text = "Some ambiguous output without clear status."
    result = parse_validator_output(text)
    assert result.validation_status == "NEEDS_REVISION"


def test_parse_yaml_block() -> None:
    text = '''Here is the result:
```yaml
validation_status: PASS
issues: []
```
'''
    result = parse_validator_output(text)
    assert result.validation_status == "PASS"
    assert result.issues == []


def test_parse_pass_without_prefix() -> None:
    text = "Everything looks good. PASS."
    result = parse_validator_output(text)
    assert result.validation_status == "PASS"


def test_validator_output_model() -> None:
    out = ValidatorOutput(
        validation_status="PASS",
        issues=[],
        final_output_override=None,
    )
    assert out.validation_status == "PASS"


def test_researcher_output_model() -> None:
    out = ResearcherOutput(
        reasoning="test",
        findings="test findings",
        sources=[],
        confidence="HIGH",
        uncertainties=[],
    )
    assert out.confidence == "HIGH"
