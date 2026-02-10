"""Pydantic models for session management."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionTurnRequest(BaseModel):
    session_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    text: str = Field(min_length=1)
    meta: Optional[Dict[str, Any]] = None
    patient_id: Optional[str] = None
    return_limit: int = Field(default=10, ge=1, le=50)


class SessionTurnResponse(BaseModel):
    session_id: str
    recent_turns: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SessionSummaryUpdate(BaseModel):
    session_id: str = Field(min_length=1)
    summary: Dict[str, Any] = Field(default_factory=dict)
    patient_id: Optional[str] = None


class SessionCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)


class SessionUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class SessionMetadata(BaseModel):
    session_id: str
    user_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: str
    last_activity: str
    message_count: int = 0
    first_message_preview: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: List[SessionMetadata]
    count: int


class SessionCountResponse(BaseModel):
    user_id: str
    count: int
    max_allowed: int = 5


class ErrorResponse(BaseModel):
    error: str
    code: str
    max_sessions: int = 5
