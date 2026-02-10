from __future__ import annotations

import os
import requests
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    import boto3
    from boto3.dynamodb.conditions import Key
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    # Create dummy classes for type hints
    class ClientError(Exception):
        pass
    class NoCredentialsError(Exception):
        pass


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ttl_epoch(ttl_days: Optional[int]) -> Optional[int]:
    if not ttl_days or ttl_days <= 0:
        return None
    return int(time.time() + ttl_days * 86400)


def _validate_table_name(table_name: str) -> str:
    """Validate DynamoDB table name according to AWS rules.

    Rules:
    - Must be between 3 and 255 characters long
    - May contain only: a-z, A-Z, 0-9, '_', '-', and '.'

    Raises ValueError if invalid.
    """
    if not table_name:
        raise ValueError("Table name cannot be empty")

    if len(table_name) < 3 or len(table_name) > 255:
        raise ValueError(
            f"Table name '{table_name}' must be between 3 and 255 characters long "
            f"(got {len(table_name)} characters)"
        )

    import re
    if not re.match(r'^[a-zA-Z0-9_.-]+$', table_name):
        raise ValueError(
            f"Table name '{table_name}' contains invalid characters. "
            "Only a-z, A-Z, 0-9, '_', '-', and '.' are allowed."
        )

    return table_name


@dataclass
class SessionTurn:
    session_id: str
    turn_ts: str
    role: str
    text: str
    meta: Dict[str, Any]
    patient_id: Optional[str] = None
    ttl: Optional[int] = None


class SessionStore:
    """
    DynamoDB-backed session store for short conversation windows and summaries.
    No external cache; uses last-N query per request.
    """

    def __init__(
        self,
        region_name: str,
        turns_table: str,
        summary_table: str,
        endpoint_url: Optional[str] = None,
        ttl_days: Optional[int] = None,
        max_recent: int = 10,
        auto_create: bool = False,
    ) -> None:
        # Validate table names before proceeding
        try:
            validated_turns_table = _validate_table_name(turns_table)
            validated_summary_table = _validate_table_name(summary_table)
        except ValueError as e:
            raise ValueError(f"Invalid DynamoDB table name: {e}") from e

        self.region_name = region_name
        self.turns_table_name = validated_turns_table
        self.summary_table_name = validated_summary_table
        self.ttl_days = ttl_days
        self.max_recent = max_recent

        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for DynamoDB session store")

        # Initialize DynamoDB resource
        # Use dummy credentials for local DynamoDB if endpoint_url is set
        if endpoint_url:
            # Local DynamoDB - use dummy credentials
            self.resource = boto3.resource(
                "dynamodb",
                region_name=region_name,
                endpoint_url=endpoint_url,
                aws_access_key_id="dummy",
                aws_secret_access_key="dummy",
            )
        else:
            # Real AWS - use default credential chain
            self.resource = boto3.resource("dynamodb", region_name=region_name)

        self.turns_table = self.resource.Table(turns_table)
        self.summary_table = self.resource.Table(summary_table)

        if auto_create:
            self.ensure_tables()

    # ------------------------ table management ------------------------ #

    def ensure_tables(self) -> None:
        """Create tables if missing (PAY_PER_REQUEST) and enable TTL if configured."""
        client = self.resource.meta.client
        self._ensure_table(
            client=client,
            table_name=self.turns_table_name,
            key_schema=[
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "turn_ts", "KeyType": "RANGE"},
            ],
            attribute_definitions=[
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "turn_ts", "AttributeType": "S"},
            ],
            ttl_attribute="ttl" if self.ttl_days and self.ttl_days > 0 else None,
        )
        self._ensure_table(
            client=client,
            table_name=self.summary_table_name,
            key_schema=[
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            attribute_definitions=[
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "last_activity", "AttributeType": "S"},
            ],
            ttl_attribute="ttl" if self.ttl_days and self.ttl_days > 0 else None,
        )
        # Create GSI for user_id queries
        self._ensure_gsi(client, self.summary_table_name)

    def _ensure_table(
        self,
        client,
        table_name: str,
        key_schema: List[Dict[str, str]],
        attribute_definitions: List[Dict[str, str]],
        ttl_attribute: Optional[str] = None,
    ) -> None:
        # Validate table name before DynamoDB operations
        try:
            _validate_table_name(table_name)
        except ValueError as e:
            raise ValueError(f"Cannot ensure table with invalid name '{table_name}': {e}") from e

        try:
            client.describe_table(TableName=table_name)
            exists = True
        except client.exceptions.ResourceNotFoundException:
            exists = False
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ValidationException':
                raise ValueError(
                    f"DynamoDB validation error for table '{table_name}': {e.response.get('Error', {}).get('Message', str(e))}. "
                    f"Table name must be 3-255 characters and contain only a-z, A-Z, 0-9, '_', '-', and '.'"
                ) from e
            raise

        if not exists:
            client.create_table(
                TableName=table_name,
                KeySchema=key_schema,
                AttributeDefinitions=attribute_definitions,
                BillingMode="PAY_PER_REQUEST",
            )
            waiter = client.get_waiter("table_exists")
            waiter.wait(TableName=table_name)

        if ttl_attribute:
            try:
                client.update_time_to_live(
                    TableName=table_name,
                    TimeToLiveSpecification={"Enabled": True, "AttributeName": ttl_attribute},
                )
            except ClientError:
                # If TTL already set or not permitted, ignore silently
                pass

    def _ensure_gsi(self, client, table_name: str) -> None:
        """Ensure GSI exists for user_id queries."""
        try:
            table_desc = client.describe_table(TableName=table_name)
            existing_indexes = {idx["IndexName"] for idx in table_desc.get("Table", {}).get("GlobalSecondaryIndexes", [])}

            if "user_id-index" not in existing_indexes:
                try:
                    client.update_table(
                        TableName=table_name,
                        AttributeDefinitions=[
                            {"AttributeName": "user_id", "AttributeType": "S"},
                            {"AttributeName": "last_activity", "AttributeType": "S"},
                        ],
                        GlobalSecondaryIndexUpdates=[
                            {
                                "Create": {
                                    "IndexName": "user_id-index",
                                    "KeySchema": [
                                        {"AttributeName": "user_id", "KeyType": "HASH"},
                                        {"AttributeName": "last_activity", "KeyType": "RANGE"},
                                    ],
                                    "Projection": {"ProjectionType": "ALL"},
                                }
                            }
                        ],
                    )
                    # Wait for index to be active
                    waiter = client.get_waiter("table_exists")
                    waiter.wait(TableName=table_name)
                except ClientError as e:
                    # Index might already exist or creation failed - log but don't fail
                    print(f"Note: Could not create GSI (may already exist): {e}")
        except ClientError as e:
            print(f"Note: Could not check/create GSI: {e}")

    # ------------------------ operations ------------------------ #

    def append_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
    ) -> SessionTurn:
        turn_ts = _utc_iso()
        ttl = _ttl_epoch(self.ttl_days)
        item: Dict[str, Any] = {
            "session_id": session_id,
            "turn_ts": turn_ts,
            "role": role,
            "text": text,
            "meta": meta or {},
        }
        if patient_id:
            item["patient_id"] = patient_id
        if ttl:
            item["ttl"] = ttl
        self.turns_table.put_item(Item=item)
        return SessionTurn(session_id=session_id, turn_ts=turn_ts, role=role, text=text, meta=item["meta"], patient_id=patient_id, ttl=ttl)

    def get_recent(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        lim = limit or self.max_recent
        resp = self.turns_table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=False,  # newest first
            Limit=lim,
        )
        items = resp.get("Items", [])
        # Return newest-first; callers can reverse if they prefer chronological.
        return items

    def update_summary(
        self,
        session_id: str,
        summary: Dict[str, Any],
        patient_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        ttl = _ttl_epoch(self.ttl_days)
        # Persist under SK=summary
        # Persist under SK=summary
        expr = ["#updated_at = :updated_at"]
        values: Dict[str, Any] = {":updated_at": _utc_iso()}
        names: Dict[str, str] = {"#updated_at": "updated_at"}

        if patient_id:
            expr.append("#patient_id = :patient_id")
            values[":patient_id"] = patient_id
            names["#patient_id"] = "patient_id"
        if user_id:
            expr.append("#user_id = :user_id")
            values[":user_id"] = user_id
            names["#user_id"] = "user_id"

        for key, val in summary.items():
            # Use expression attribute names to handle reserved words like 'name', 'status'
            expr.append(f"#{key} = :{key}")
            values[f":{key}"] = val
            names[f"#{key}"] = key

        if ttl:
            expr.append("#ttl = :ttl")
            values[":ttl"] = ttl
            names["#ttl"] = "ttl"

        update_expr = "SET " + ", ".join(expr)
        self.summary_table.update_item(
            Key={"session_id": session_id, "sk": "summary"},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=values,
            ExpressionAttributeNames=names,
        )

    def get_summary(self, session_id: str) -> Dict[str, Any]:
        resp = self.summary_table.get_item(Key={"session_id": session_id, "sk": "summary"})
        return resp.get("Item", {})

    def set_patient(self, session_id: str, patient_id: str) -> None:
        self.update_summary(session_id=session_id, summary={}, patient_id=patient_id)

    def get_patient(self, session_id: str) -> Optional[str]:
        item = self.get_summary(session_id)
        return item.get("patient_id")

    def clear_session(self, session_id: str) -> None:
        """Delete all turns and summary for a session."""
        # Delete summary
        try:
            self.summary_table.delete_item(Key={"session_id": session_id, "sk": "summary"})
        except ClientError:
            pass
        # Delete turns in batches
        resp = self.turns_table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ProjectionExpression="session_id, turn_ts",
        )
        items = resp.get("Items", [])
        while True:
            with self.turns_table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(Key={"session_id": item["session_id"], "turn_ts": item["turn_ts"]})
            if "LastEvaluatedKey" not in resp:
                break
            resp = self.turns_table.query(
                KeyConditionExpression=Key("session_id").eq(session_id),
                ProjectionExpression="session_id, turn_ts",
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
            items = resp.get("Items", [])

    def list_sessions_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """List all sessions for a user, sorted by last_activity (newest first)."""
        items = []
        try:
            # Try to use GSI if available
            start_key = None
            while True:
                kwargs = {
                    "IndexName": "user_id-index",
                    "KeyConditionExpression": Key("user_id").eq(user_id),
                    "FilterExpression": Key("sk").eq("summary"),
                    "ScanIndexForward": False,  # Sort descending by last_activity
                }
                if start_key:
                    kwargs["ExclusiveStartKey"] = start_key

                resp = self.summary_table.query(**kwargs)
                items.extend(resp.get("Items", []))

                start_key = resp.get("LastEvaluatedKey")
                if not start_key:
                    break
        except ClientError:
            # Fallback to scan with filter if GSI not available
            start_key = None
            while True:
                kwargs = {
                    "FilterExpression": Key("sk").eq("summary") & Key("user_id").eq(user_id),
                }
                if start_key:
                    kwargs["ExclusiveStartKey"] = start_key

                resp = self.summary_table.scan(**kwargs)
                items.extend(resp.get("Items", []))

                start_key = resp.get("LastEvaluatedKey")
                if not start_key:
                    break

        # Sort by last_activity if not already sorted
        items.sort(key=lambda x: x.get("last_activity", x.get("updated_at", "")), reverse=True)
        return items

    def get_session_count(self, user_id: str) -> int:
        """Count sessions for a user."""
        try:
            # Try to use GSI if available
            resp = self.summary_table.query(
                IndexName="user_id-index",
                KeyConditionExpression=Key("user_id").eq(user_id),
                FilterExpression=Key("sk").eq("summary"),
                Select="COUNT",
            )
            return resp.get("Count", 0)
        except ClientError:
            # Fallback to scan with filter
            resp = self.summary_table.scan(
                FilterExpression=Key("sk").eq("summary") & Key("user_id").eq(user_id),
                Select="COUNT",
            )
            return resp.get("Count", 0)

    def get_first_message_preview(self, session_id: str, max_length: int = 100) -> Optional[str]:
        """Get first user message for preview."""
        try:
            resp = self.turns_table.query(
                KeyConditionExpression=Key("session_id").eq(session_id),
                FilterExpression=Key("role").eq("user"),
                ScanIndexForward=True,  # Oldest first
                Limit=1,
            )
            items = resp.get("Items", [])
            if items:
                text = items[0].get("text", "")
                if len(text) > max_length:
                    return text[:max_length] + "..."
                return text
        except ClientError:
            pass
        return None

    def migrate_existing_sessions(self, default_user_id: str) -> int:
        """Assign default user_id to existing sessions without one. Returns count migrated."""
        migrated = 0
        try:
            # Scan all summaries
            resp = self.summary_table.scan(
                FilterExpression=Key("sk").eq("summary"),
            )
            items = resp.get("Items", [])

            for item in items:
                if "user_id" not in item:
                    session_id = item.get("session_id")
                    if session_id:
                        # Update summary with default user_id
                        self.update_summary(
                            session_id=session_id,
                            summary=item.get("summary", {}),
                            patient_id=item.get("patient_id"),
                            user_id=default_user_id,
                        )
                        migrated += 1

            # Handle pagination
            while "LastEvaluatedKey" in resp:
                resp = self.summary_table.scan(
                    FilterExpression=Key("sk").eq("summary"),
                    ExclusiveStartKey=resp["LastEvaluatedKey"],
                )
                items = resp.get("Items", [])
                for item in items:
                    if "user_id" not in item:
                        session_id = item.get("session_id")
                        if session_id:
                            self.update_summary(
                                session_id=session_id,
                                summary=item.get("summary", {}),
                                patient_id=item.get("patient_id"),
                                user_id=default_user_id,
                            )
                            migrated += 1
        except ClientError as e:
            print(f"Error migrating sessions: {e}")

        return migrated


def _clean_table_name(name: str) -> str:
    """Clean and normalize table name from environment variable.

    Removes:
    - Leading/trailing whitespace
    - Quotes (single or double)
    - Common invalid prefixes like "default "
    """
    if not name:
        return name

    # Strip whitespace and quotes
    name = name.strip().strip('"').strip("'")

    # Remove common invalid prefixes (case-insensitive)
    prefixes_to_remove = ["default ", "default-", "default_"]
    for prefix in prefixes_to_remove:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix):]
            break

    return name.strip()


def _warn_on_bad_endpoint(endpoint: str) -> None:
    if not endpoint:
        return
    if "localhost:8000" in endpoint or "127.0.0.1:8000" in endpoint:
        try:
            resp = requests.get("http://localhost:8000/agent/health", timeout=2)
            if resp.status_code == 200:
                print(
                    "Warning: DDB_ENDPOINT points to FastAPI on port 8000. "
                    "DynamoDB Local should run on port 8001."
                )
        except requests.RequestException:
            print(
                "Warning: DDB_ENDPOINT appears to use port 8000. "
                "If this is DynamoDB Local, update to http://localhost:8001."
            )


# Global singleton session store instance
_SESSION_STORE: Optional[SessionStore] = None


def build_store_from_env() -> SessionStore:
    """Factory that builds a SessionStore using environment variables.

    Note: For better performance and to avoid connection pool exhaustion,
    use get_session_store() instead, which returns a singleton instance.
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    turns_table_raw = os.getenv("DDB_TURNS_TABLE", "hcai_session_turns")
    summary_table_raw = os.getenv("DDB_SUMMARY_TABLE", "hcai_session_summary")

    # Clean table names
    turns_table = _clean_table_name(turns_table_raw)
    summary_table = _clean_table_name(summary_table_raw)

    endpoint = os.getenv("DDB_ENDPOINT", "http://localhost:8001")  # Default to local DynamoDB
    _warn_on_bad_endpoint(endpoint)
    ttl_days_str = os.getenv("DDB_TTL_DAYS")
    ttl_days = int(ttl_days_str) if ttl_days_str and ttl_days_str.isdigit() else None
    auto_create = os.getenv("DDB_AUTO_CREATE", "false").lower() in {"1", "true", "yes"}
    max_recent = int(os.getenv("SESSION_RECENT_LIMIT", "10"))

    # Validate table names before creating store
    try:
        _validate_table_name(turns_table)
        _validate_table_name(summary_table)
    except ValueError as e:
        raise ValueError(f"Invalid table name in environment variables: {e}") from e

    return SessionStore(
        region_name=region,
        turns_table=turns_table,
        summary_table=summary_table,
        endpoint_url=endpoint,
        ttl_days=ttl_days,
        max_recent=max_recent,
        auto_create=auto_create,
    )


def get_session_store() -> SessionStore:
    """Get or create singleton session store instance.

    This function returns a singleton SessionStore instance to avoid
    creating multiple DynamoDB connections, which can cause connection
    pool exhaustion and timeouts under concurrent load.

    Returns:
        SessionStore: Singleton session store instance
    """
    global _SESSION_STORE
    if _SESSION_STORE is None:
        _SESSION_STORE = build_store_from_env()
    return _SESSION_STORE
