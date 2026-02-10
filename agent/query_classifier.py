"""Query classifier for routing between conversational and medical queries.

Uses a hybrid approach:
1. Fast keyword matching for obvious cases (<100ms)
2. LLM fallback for ambiguous queries

Query Types:
- CONVERSATIONAL: Greetings, meta questions, social queries
- MEDICAL: Clinical questions, patient data, calculations
- MIXED: Both greeting + medical intent (routes to medical with acknowledgment)
- UNCLEAR: Ambiguous, uses LLM or session context
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

class QueryType(Enum):
    CONVERSATIONAL = "conversational"
    MEDICAL = "medical"
    MIXED = "mixed"
    UNCLEAR = "unclear"


@dataclass
class ClassificationResult:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    method: str  # "keyword" | "context" | "llm_fallback"
    matched_keywords: List[str]
    should_acknowledge_greeting: bool = False
    processing_time_ms: float = 0.0


# Conversational patterns - things that indicate chat, not medical
CONVERSATIONAL_PATTERNS = {
    # Greetings
    "greetings": [
        r"\b(hello|hi|hey|greetings|howdy)\b",
        r"\bgood\s+(morning|afternoon|evening|day)\b",
        r"\b(goodbye|bye|see\s+you|farewell|take\s+care)\b",
    ],
    # Identity questions
    "identity": [
        r"\bwho\s+are\s+you\b",
        r"\bwhat\s+are\s+you\b",
        r"\byour\s+name\b",
        r"\bintroduce\s+yourself\b",
        r"\bare\s+you\s+(a\s+|an\s+)?(bot|ai|robot|assistant|doctor|human)\b",
    ],
    # Capability questions
    "capability": [
        r"\bwhat\s+can\s+you\s+do\b",
        r"\bhow\s+can\s+you\s+help\b",
        r"\bwhat\s+do\s+you\s+know\b",
        r"\bwhat\s+are\s+your\s+capabilities\b",
        r"\bhelp\s+me\b",
    ],
    # Social
    "social": [
        r"\bhow\s+are\s+you\b",
        r"\b(thank\s+you|thanks|thx)\b",
        r"\bplease\b$",  # just "please" at end
        r"\b(sorry|excuse\s+me)\b",
    ],
}

# Medical patterns - things that indicate clinical/healthcare intent
MEDICAL_PATTERNS = {
    # Patient-specific
    "patient": [
        r"\bpatient\s+[a-zA-Z0-9\-]+\b",
        r"\bpatient_id\b",
        r"\bcheck\s+patient\b",
        r"\b(his|her|their)\s+(medication|diagnosis|condition|labs?|records?)\b",
    ],
    # Clinical terms
    "clinical": [
        r"\b(diagnosis|symptom|symptoms|treatment|therapy|prognosis)\b",
        r"\b(medication|medications|medicine|drug|drugs|prescription|dose|dosage)\b",
        r"\b(lab|labs|laboratory|test|tests|blood\s+test)\b",
        r"\b(condition|disorder|syndrome|disease|illness)\b",
    ],
    # Measurements
    "measurements": [
        r"\b(blood\s+pressure|bp)\b",
        r"\b(glucose|sugar\s+level|a1c|hba1c)\b",
        r"\b(heart\s+rate|pulse|hr)\b",
        r"\b(temperature|fever)\b",
        r"\b(weight|height|bmi|bsa)\b",
        r"\b(gfr|egfr|creatinine\s+clearance|crcl)\b",
    ],
    # Conditions
    "conditions": [
        r"\b(diabetes|diabetic)\b",
        r"\b(hypertension|high\s+blood\s+pressure)\b",
        r"\b(cancer|tumor|malignant)\b",
        r"\b(infection|bacterial|viral)\b",
        r"\b(heart\s+disease|cardiovascular)\b",
        r"\b(kidney|renal|hepatic|liver)\b",
    ],
    # Actions
    "actions": [
        r"\bcalculate\b",
        r"\banalyze\b",
        r"\bassess\b",
        r"\bevaluate\b",
        r"\bvalidate\b",
        r"\bcheck\b(?!\s+out)",  # "check" but not "check out"
        r"\blook\s+up\b",
        r"\bsearch\b",
    ],
    # Codes
    "codes": [
        r"\bicd-?10\b",
        r"\bloinc\b",
        r"\brxnorm\b",
        r"\bsnomed\b",
        r"\bcpt\b",
        r"\b[A-Z]\d{2}(\.\d+)?\b",  # ICD-10 pattern like E11.9
    ],
}

# Mixed greeting patterns - greetings that often precede medical queries
MIXED_GREETING_PATTERNS = [
    r"^(hello|hi|hey|good\s+(morning|afternoon|evening))[!,.]?\s+",
]


class QueryClassifier:
    """Hybrid query classifier for medical vs conversational routing."""

    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialize classifier.

        Args:
            confidence_threshold: Minimum confidence before LLM fallback
        """
        self.confidence_threshold = confidence_threshold
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        self._conversational_compiled = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in CONVERSATIONAL_PATTERNS.items()
        }
        self._medical_compiled = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in MEDICAL_PATTERNS.items()
        }
        self._mixed_greeting_compiled = [
            re.compile(p, re.IGNORECASE) for p in MIXED_GREETING_PATTERNS
        ]

    def classify(
        self,
        query: str,
        session_context: Optional[dict] = None,
        use_llm_fallback: bool = True
    ) -> ClassificationResult:
        """
        Classify a query as conversational, medical, mixed, or unclear.

        Args:
            query: The user's query text
            session_context: Optional session state for context-aware classification
            use_llm_fallback: Whether to use LLM for ambiguous cases

        Returns:
            ClassificationResult with type, confidence, and metadata
        """
        start_time = time.time()

        query = query.strip()

        # Handle empty queries
        if not query:
            return ClassificationResult(
                query_type=QueryType.UNCLEAR,
                confidence=1.0,
                method="empty",
                matched_keywords=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Step 1: Keyword matching
        conv_matches = self._match_patterns(query, self._conversational_compiled)
        med_matches = self._match_patterns(query, self._medical_compiled)

        # Check for mixed greeting prefix
        has_greeting_prefix = any(p.match(query) for p in self._mixed_greeting_compiled)

        # Step 2: Determine classification
        conv_score = len(conv_matches)
        med_score = len(med_matches)

        # Case: Mixed - greeting + medical content
        if has_greeting_prefix and med_score > 0:
            result = ClassificationResult(
                query_type=QueryType.MIXED,
                confidence=0.9,
                method="keyword",
                matched_keywords=med_matches,
                should_acknowledge_greeting=True,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Case: Pure conversational - only conversational keywords
        elif conv_score > 0 and med_score == 0:
            result = ClassificationResult(
                query_type=QueryType.CONVERSATIONAL,
                confidence=min(0.95, 0.7 + conv_score * 0.1),
                method="keyword",
                matched_keywords=conv_matches,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Case: Pure medical - only medical keywords
        elif med_score > 0 and conv_score == 0:
            result = ClassificationResult(
                query_type=QueryType.MEDICAL,
                confidence=min(0.95, 0.7 + med_score * 0.1),
                method="keyword",
                matched_keywords=med_matches,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Case: Both - more medical wins (treat as medical)
        elif med_score > conv_score:
            result = ClassificationResult(
                query_type=QueryType.MEDICAL,
                confidence=0.7,
                method="keyword",
                matched_keywords=med_matches,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Case: Both - more conversational wins (but might be medical context)
        elif conv_score > med_score:
            result = ClassificationResult(
                query_type=QueryType.CONVERSATIONAL,
                confidence=0.7,
                method="keyword",
                matched_keywords=conv_matches,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Case: No matches - UNCLEAR
        else:
            # Try session context first
            if session_context and session_context.get("last_query_type"):
                last_type = session_context["last_query_type"]
                result = ClassificationResult(
                    query_type=QueryType.MEDICAL if last_type == "medical" else QueryType.CONVERSATIONAL,
                    confidence=0.6,
                    method="context",
                    matched_keywords=[],
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            else:
                result = ClassificationResult(
                    query_type=QueryType.UNCLEAR,
                    confidence=0.5,
                    method="keyword",
                    matched_keywords=[],
                    processing_time_ms=(time.time() - start_time) * 1000
                )

        # Step 3: LLM fallback for low confidence or unclear
        if result.confidence < self.confidence_threshold and result.query_type == QueryType.UNCLEAR:
            if use_llm_fallback:
                result = self._llm_classify(query, result, start_time)
            else:
                # Default to medical (safer)
                result.query_type = QueryType.MEDICAL
                result.method = "default"

        return result

    def _match_patterns(
        self,
        query: str,
        pattern_dict: dict
    ) -> List[str]:
        """Match query against a set of compiled patterns."""
        matches = []
        for category, patterns in pattern_dict.items():
            for pattern in patterns:
                if pattern.search(query):
                    matches.append(f"{category}:{pattern.pattern}")
        return matches

    def _llm_classify(
        self,
        query: str,
        current_result: ClassificationResult,
        start_time: float
    ) -> ClassificationResult:
        """
        Use LLM for classification when keywords are ambiguous.

        Note: This is a placeholder - actual implementation would call the LLM.
        For now, defaults to MEDICAL (safer fallback).
        """
        # TODO: Implement actual LLM classification
        # For now, default to MEDICAL for safety
        return ClassificationResult(
            query_type=QueryType.MEDICAL,
            confidence=0.7,
            method="llm_fallback_default",
            matched_keywords=current_result.matched_keywords,
            processing_time_ms=(time.time() - start_time) * 1000
        )


# Convenience function for quick classification
def classify_query(
    query: str,
    session_context: Optional[dict] = None
) -> ClassificationResult:
    """
    Classify a query using the default classifier.

    Args:
        query: The user's query text
        session_context: Optional session state

    Returns:
        ClassificationResult
    """
    classifier = QueryClassifier()
    return classifier.classify(query, session_context)
