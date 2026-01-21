"""Session store for conversation history and summaries.

Supports both in-memory storage (default) and DynamoDB (when configured).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from logging_config import get_logger

load_dotenv()

logger = get_logger("hc_ai.session")

# Session configuration
SESSION_PROVIDER = os.getenv("SESSION_PROVIDER", "memory").lower()
SESSION_RECENT_LIMIT = int(os.getenv("SESSION_RECENT_LIMIT", "10"))
SESSION_TTL_DAYS = os.getenv("SESSION_TTL_DAYS", "")
SESSION_MAX_SESSIONS = int(os.getenv("SESSION_MAX_SESSIONS", "1000"))
SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "300"))

# DynamoDB configuration (optional)
DDB_REGION = os.getenv("AWS_REGION", "us-east-1")
DDB_TURNS_TABLE = os.getenv("DDB_TURNS_TABLE", "hcai_session_turns")
DDB_SUMMARY_TABLE = os.getenv("DDB_SUMMARY_TABLE", "hcai_session_summary")
DDB_ENDPOINT = os.getenv("DDB_ENDPOINT")
DDB_AUTO_CREATE = os.getenv("DDB_AUTO_CREATE", "false").lower() in {"1", "true", "yes"}


def _utc_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ttl_epoch(ttl_days: Optional[int]) -> Optional[int]:
    """Calculate TTL epoch timestamp."""
    if not ttl_days or ttl_days <= 0:
        return None
    return int(time.time() + ttl_days * 86400)


@dataclass
class SessionTurn:
    """A single turn in a conversation."""
    
    session_id: str
    turn_ts: str
    role: str
    text: str
    meta: Dict[str, Any]
    patient_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "turn_ts": self.turn_ts,
            "role": self.role,
            "text": self.text,
            "meta": self.meta,
        }
        if self.patient_id:
            result["patient_id"] = self.patient_id
        return result


class SessionStore:
    """Session store for conversation history.
    
    Supports in-memory storage (default) or DynamoDB.
    """
    
    def __init__(
        self,
        provider: str = SESSION_PROVIDER,
        max_recent: int = SESSION_RECENT_LIMIT,
        ttl_days: Optional[int] = None,
    ) -> None:
        """Initialize the session store.
        
        Args:
            provider: Storage provider ("memory" or "dynamodb").
            max_recent: Maximum recent turns to return.
            ttl_days: TTL in days for session data (DynamoDB only).
        """
        self.provider = provider
        self.max_recent = max_recent
        
        env_ttl_days = os.getenv("SESSION_TTL_DAYS", SESSION_TTL_DAYS)
        if env_ttl_days and env_ttl_days.isdigit():
            self.ttl_days = int(env_ttl_days)
        else:
            self.ttl_days = ttl_days

        env_max_sessions = os.getenv("SESSION_MAX_SESSIONS", str(SESSION_MAX_SESSIONS))
        self._max_sessions = int(env_max_sessions) if env_max_sessions.isdigit() else SESSION_MAX_SESSIONS

        env_cleanup_interval = os.getenv(
            "SESSION_CLEANUP_INTERVAL_SECONDS",
            str(SESSION_CLEANUP_INTERVAL_SECONDS),
        )
        if env_cleanup_interval.isdigit():
            self._cleanup_interval_seconds = int(env_cleanup_interval)
        else:
            self._cleanup_interval_seconds = SESSION_CLEANUP_INTERVAL_SECONDS
        
        # In-memory storage
        self._turns: Dict[str, List[Dict[str, Any]]] = {}
        self._summaries: Dict[str, Dict[str, Any]] = {}
        self._last_access: Dict[str, float] = {}
        self._last_cleanup = 0.0
        
        # DynamoDB tables (lazy initialized)
        self._ddb_resource = None
        self._turns_table = None
        self._summary_table = None
    
    def _init_dynamodb(self) -> None:
        """Initialize DynamoDB tables."""
        if self._ddb_resource is not None:
            return
        
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as e:
            raise ImportError(
                "boto3 is required for DynamoDB. Install with: pip install boto3"
            ) from e
        
        self._ddb_resource = boto3.resource(
            "dynamodb",
            region_name=DDB_REGION,
            endpoint_url=DDB_ENDPOINT,
        )
        self._turns_table = self._ddb_resource.Table(DDB_TURNS_TABLE)
        self._summary_table = self._ddb_resource.Table(DDB_SUMMARY_TABLE)
        
        if DDB_AUTO_CREATE:
            self._ensure_tables()
    
    def _ensure_tables(self) -> None:
        """Create DynamoDB tables if they don't exist."""
        client = self._ddb_resource.meta.client
        
        # Create turns table
        try:
            client.describe_table(TableName=DDB_TURNS_TABLE)
        except client.exceptions.ResourceNotFoundException:
            client.create_table(
                TableName=DDB_TURNS_TABLE,
                KeySchema=[
                    {"AttributeName": "session_id", "KeyType": "HASH"},
                    {"AttributeName": "turn_ts", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "session_id", "AttributeType": "S"},
                    {"AttributeName": "turn_ts", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            waiter = client.get_waiter("table_exists")
            waiter.wait(TableName=DDB_TURNS_TABLE)
        
        # Create summary table
        try:
            client.describe_table(TableName=DDB_SUMMARY_TABLE)
        except client.exceptions.ResourceNotFoundException:
            client.create_table(
                TableName=DDB_SUMMARY_TABLE,
                KeySchema=[
                    {"AttributeName": "session_id", "KeyType": "HASH"},
                    {"AttributeName": "sk", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "session_id", "AttributeType": "S"},
                    {"AttributeName": "sk", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            waiter = client.get_waiter("table_exists")
            waiter.wait(TableName=DDB_SUMMARY_TABLE)
    
    def append_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
    ) -> SessionTurn:
        """Append a turn to the session history.
        
        Args:
            session_id: Session identifier.
            role: Role of the speaker (e.g., "user", "assistant").
            text: Text content of the turn.
            meta: Additional metadata.
            patient_id: Optional patient ID.
        
        Returns:
            The created SessionTurn.
        """
        turn_ts = _utc_iso()
        turn = SessionTurn(
            session_id=session_id,
            turn_ts=turn_ts,
            role=role,
            text=text,
            meta=meta or {},
            patient_id=patient_id,
        )
        
        if self.provider == "dynamodb":
            self._init_dynamodb()
            item: Dict[str, Any] = turn.to_dict()
            ttl = _ttl_epoch(self.ttl_days)
            if ttl:
                item["ttl"] = ttl
            self._turns_table.put_item(Item=item)
        else:
            # In-memory storage
            self._cleanup_if_needed()
            if session_id not in self._turns:
                self._turns[session_id] = []
            self._turns[session_id].append(turn.to_dict())
            self._last_access[session_id] = time.time()
            # Trim to max_recent * 2 to allow some buffer
            if len(self._turns[session_id]) > self.max_recent * 2:
                self._turns[session_id] = self._turns[session_id][-self.max_recent * 2:]
            self._enforce_session_limit()
        
        return turn
    
    def get_recent(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent turns for a session.
        
        Args:
            session_id: Session identifier.
            limit: Maximum number of turns to return.
        
        Returns:
            List of turn dictionaries (newest first).
        """
        lim = limit or self.max_recent
        
        if self.provider == "dynamodb":
            self._init_dynamodb()
            from boto3.dynamodb.conditions import Key
            
            resp = self._turns_table.query(
                KeyConditionExpression=Key("session_id").eq(session_id),
                ScanIndexForward=False,
                Limit=lim,
            )
            return resp.get("Items", [])
        else:
            self._cleanup_if_needed()
            turns = self._turns.get(session_id, [])
            if turns:
                self._last_access[session_id] = time.time()
            # Return newest first
            return list(reversed(turns[-lim:]))
    
    def update_summary(
        self,
        session_id: str,
        summary: Dict[str, Any],
        patient_id: Optional[str] = None,
    ) -> None:
        """Update the session summary.
        
        Args:
            session_id: Session identifier.
            summary: Summary data to store.
            patient_id: Optional patient ID.
        """
        if self.provider == "dynamodb":
            self._init_dynamodb()
            
            expr_parts = ["updated_at = :updated_at"]
            values: Dict[str, Any] = {":updated_at": _utc_iso()}
            
            if patient_id:
                expr_parts.append("patient_id = :patient_id")
                values[":patient_id"] = patient_id
            
            for key, val in summary.items():
                expr_parts.append(f"{key} = :{key}")
                values[f":{key}"] = val
            
            ttl = _ttl_epoch(self.ttl_days)
            if ttl:
                expr_parts.append("ttl = :ttl")
                values[":ttl"] = ttl
            
            self._summary_table.update_item(
                Key={"session_id": session_id, "sk": "summary"},
                UpdateExpression="SET " + ", ".join(expr_parts),
                ExpressionAttributeValues=values,
            )
        else:
            self._cleanup_if_needed()
            if session_id not in self._summaries:
                self._summaries[session_id] = {}
            self._summaries[session_id].update(summary)
            self._summaries[session_id]["updated_at"] = _utc_iso()
            if patient_id:
                self._summaries[session_id]["patient_id"] = patient_id
            self._last_access[session_id] = time.time()
            self._enforce_session_limit()
    
    def get_summary(self, session_id: str) -> Dict[str, Any]:
        """Get the session summary.
        
        Args:
            session_id: Session identifier.
        
        Returns:
            Summary dictionary.
        """
        if self.provider == "dynamodb":
            self._init_dynamodb()
            resp = self._summary_table.get_item(
                Key={"session_id": session_id, "sk": "summary"}
            )
            return resp.get("Item", {})
        else:
            self._cleanup_if_needed()
            return self._summaries.get(session_id, {})
    
    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session.
        
        Args:
            session_id: Session identifier.
        """
        if self.provider == "dynamodb":
            self._init_dynamodb()
            from boto3.dynamodb.conditions import Key
            from botocore.exceptions import ClientError
            
            # Delete summary
            try:
                self._summary_table.delete_item(
                    Key={"session_id": session_id, "sk": "summary"}
                )
            except ClientError:
                pass
            
            # Delete turns in batches
            resp = self._turns_table.query(
                KeyConditionExpression=Key("session_id").eq(session_id),
                ProjectionExpression="session_id, turn_ts",
            )
            items = resp.get("Items", [])
            
            while True:
                with self._turns_table.batch_writer() as batch:
                    for item in items:
                        batch.delete_item(
                            Key={
                                "session_id": item["session_id"],
                                "turn_ts": item["turn_ts"],
                            }
                        )
                
                if "LastEvaluatedKey" not in resp:
                    break
                
                resp = self._turns_table.query(
                    KeyConditionExpression=Key("session_id").eq(session_id),
                    ProjectionExpression="session_id, turn_ts",
                    ExclusiveStartKey=resp["LastEvaluatedKey"],
                )
                items = resp.get("Items", [])
        else:
            self._turns.pop(session_id, None)
            self._summaries.pop(session_id, None)
            self._last_access.pop(session_id, None)

    def _cleanup_if_needed(self) -> None:
        """Cleanup stale in-memory sessions based on TTL and interval."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval_seconds:
            return
        self._last_cleanup = now

        if not self.ttl_days or self.ttl_days <= 0:
            return

        ttl_seconds = self.ttl_days * 86400
        expired = [sid for sid, ts in self._last_access.items() if now - ts > ttl_seconds]
        for sid in expired:
            self._turns.pop(sid, None)
            self._summaries.pop(sid, None)
            self._last_access.pop(sid, None)
        if expired:
            logger.info("Cleaned up %s expired sessions", len(expired))

    def _enforce_session_limit(self) -> None:
        """Enforce a max session limit in memory."""
        if self._max_sessions <= 0:
            return
        if len(self._last_access) <= self._max_sessions:
            return
        # Evict least recently used sessions
        sorted_sessions = sorted(self._last_access.items(), key=lambda item: item[1])
        to_evict = len(self._last_access) - self._max_sessions
        for sid, _ts in sorted_sessions[:to_evict]:
            self._turns.pop(sid, None)
            self._summaries.pop(sid, None)
            self._last_access.pop(sid, None)
        logger.info("Evicted %s sessions to enforce limit", to_evict)


# Global session store instance
_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Get or create the global session store instance."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
