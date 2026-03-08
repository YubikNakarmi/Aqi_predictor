import json
import os
from datetime import datetime, timezone
from typing import Any


def _json_default(value: Any):# Convert datetime to ISO format, and handle other non-serializable types
    if isinstance(value, (datetime,)):
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _load_json(file_path: str) -> dict: #load joson 
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return {}


def _write_json(file_path: str, data: dict) -> None: #write to file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=_json_default)


def update_pipeline_metadata(metadata_file: str, run_payload: dict, max_history: int = 30) -> None:
    data = _load_json(metadata_file) #original file
    data["last_updated_utc"] = datetime.now(timezone.utc).isoformat()#set and update datetime

    runtime = data.get("runtime", {})#get runtime info for current run
    runtime["last_run"] = run_payload

    history = runtime.get("history", [])#appending rolling history each run
    history.append(run_payload)
    runtime["history"] = history[-max_history:]

    data["runtime"] = runtime
    _write_json(metadata_file, data)
