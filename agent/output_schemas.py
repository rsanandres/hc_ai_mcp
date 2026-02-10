"""Pydantic schemas for structured agent output parsing."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """A single issue found during validation."""

    description: str = Field(..., description="Description of the issue")
    severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Severity level of the issue"
    )
    fix_required: str = Field(..., description="What the researcher needs to fix")


class ValidatorOutput(BaseModel):
    """Structured output from the validator agent."""

    validation_status: Literal["PASS", "NEEDS_REVISION", "FAIL"] = Field(
        ..., description="Overall validation result"
    )
    issues: List[ValidationIssue] = Field(
        default_factory=list, description="List of issues found"
    )
    final_output_override: Optional[str] = Field(
        None, description="Corrected output to use instead of researcher output"
    )


class ResearcherOutput(BaseModel):
    """Structured output from the researcher agent."""

    reasoning: str = Field(..., description="Brief reasoning about approach")
    findings: str = Field(..., description="The clinical answer")
    sources: List[str] = Field(
        default_factory=list, description="Citations in format [FHIR:Type/ID]"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Confidence level in the findings"
    )
    uncertainties: List[str] = Field(
        default_factory=list, description="Data gaps or limitations"
    )


def parse_validator_output(text: str) -> ValidatorOutput:
    """
    Parse validator output from text, with fallback handling.

    Attempts YAML parsing first, then falls back to text parsing.
    """
    import re
    import yaml

    # Try to extract YAML block
    yaml_match = re.search(r"```yaml\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if yaml_match:
        try:
            parsed = yaml.safe_load(yaml_match.group(1))
            return ValidatorOutput(**parsed)
        except Exception:
            pass  # Fall through to text parsing

    # Try raw YAML parsing (without code fences)
    try:
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict) and "validation_status" in parsed:
            return ValidatorOutput(**parsed)
    except Exception:
        pass

    # Fallback: text parsing
    validation_status: Literal["PASS", "NEEDS_REVISION", "FAIL"] = "NEEDS_REVISION"

    text_upper = text.upper()
    if "VALIDATION_STATUS:" in text_upper or "VALIDATION_STATUS :" in text_upper:
        # Look for status after the colon
        for line in text.splitlines():
            line_upper = line.upper().strip()
            if "VALIDATION_STATUS" in line_upper:
                if "PASS" in line_upper:
                    validation_status = "PASS"
                elif "FAIL" in line_upper:
                    validation_status = "FAIL"
                break
    elif "PASS" in text_upper and "FAIL" not in text_upper:
        validation_status = "PASS"
    elif "FAIL" in text_upper:
        validation_status = "FAIL"

    return ValidatorOutput(
        validation_status=validation_status,
        issues=[],
        final_output_override=None,
    )
