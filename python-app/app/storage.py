"""Local fallback storage for analyses.

Used when MongoDB is unavailable so the app can still analyze images and
serve history/statistics from a writable filesystem store.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime

from flask import current_app

_STORE_LOCK = threading.Lock()
_STORE_FILENAME = "analysis_store.json"


def _store_path() -> str:
    os.makedirs(current_app.instance_path, exist_ok=True)
    return os.path.join(current_app.instance_path, _STORE_FILENAME)


def _read_records() -> list[dict]:
    path = _store_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_records(records: list[dict]) -> None:
    path = _store_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def save_local_analysis(doc: dict) -> dict:
    """Insert or replace an analysis record in the local store."""
    record = dict(doc)
    record.setdefault("_id", uuid.uuid4().hex)
    now = datetime.now().astimezone().isoformat()
    created_at = record.get("created_at", now)
    if isinstance(created_at, datetime):
        created_at = created_at.astimezone().isoformat()
    record["created_at"] = created_at
    record["updated_at"] = now

    with _STORE_LOCK:
        records = _read_records()
        records = [existing for existing in records if str(existing.get("_id")) != str(record["_id"])]
        records.insert(0, record)
        _write_records(records)

    return record


def list_local_analyses() -> list[dict]:
    return _read_records()


def get_local_analysis(record_id: str) -> dict | None:
    for record in _read_records():
        if str(record.get("_id")) == str(record_id):
            return record
    return None


def delete_local_analysis(record_id: str) -> dict | None:
    with _STORE_LOCK:
        records = _read_records()
        remaining = [record for record in records if str(record.get("_id")) != str(record_id)]
        deleted = next((record for record in records if str(record.get("_id")) == str(record_id)), None)
        if deleted is not None:
            _write_records(remaining)
        return deleted
