# controller.py
# Agentest Controller (v1, no-ODS) + Snapshots & Logs
# - GET  /healthz
# - POST /state   : fetch raw SUT state -> filter -> return (and log both)
# - POST /action  : guard -> forward to SUT (and log plan + reasoning + result)
#
# Env:
#   REAL_SUT_STATE=http://127.0.0.1:18080/state
#   REAL_SUT_ACTION=http://127.0.0.1:18080/action
#   LOG_DIR=./logs
#   CONTROLLER_LOG_SNAPSHOTS=1   # 0 kapatır
#   CONTROLLER_LOG_THOUGHTS=1    # 0 kapatır
#
# Run:
#   uvicorn controller:app --host 0.0.0.0 --port 20080 --reload

from __future__ import annotations
import os
import time
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from filter_utils import filter_pipeline_v1, guard_action_plan_v1

REAL_SUT_STATE = os.getenv("REAL_SUT_STATE", "http://127.0.0.1:18080/state")
REAL_SUT_ACTION = os.getenv("REAL_SUT_ACTION", "http://127.0.0.1:18080/action")

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs")).resolve()
LOG_SNAPSHOTS = os.getenv("CONTROLLER_LOG_SNAPSHOTS", "1") != "0"
LOG_THOUGHTS = os.getenv("CONTROLLER_LOG_THOUGHTS", "1") != "0"

# prepare log folders
STATE_DIR = LOG_DIR / "state"
ACTION_DIR = LOG_DIR / "action"
REASON_DIR = LOG_DIR / "reasoning"
for d in (STATE_DIR, ACTION_DIR, REASON_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Agentest Controller (v1, no-ODS)", version="0.2.0")
session = requests.Session()

class StateRequest(BaseModel):
    # optional hint (step/expected) → filter scoring’inde kullanılabilir
    hint: Optional[str] = Field(default=None, description="Optional step text to bias filtering")

def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _snap_name(prefix: str) -> str:
    # ör: raw_20250101-103000_5f3a2.json
    return f"{prefix}_{_ts()}_{uuid.uuid4().hex[:5]}.json"

def _write_json(path: Path, data: Any) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        # logging fail should not break API
        pass

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "sut_state": REAL_SUT_STATE,
        "sut_action": REAL_SUT_ACTION,
        "log_dir": str(LOG_DIR),
        "snapshots": LOG_SNAPSHOTS,
        "thoughts": LOG_THOUGHTS,
        "ts": int(time.time()),
    }

@app.post("/state")
def state_proxy(payload: StateRequest = Body(default=StateRequest())):
    """
    Fetch SUT /state → filter → return.
    Logs:
      - logs/state/raw_*.json
      - logs/state/filtered_*.json
    """
    # 1) ham state’i al
    try:
        r = session.post(REAL_SUT_STATE, json={}, timeout=15)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        raise HTTPException(502, f"SUT /state failed: {e}")

    # snapshot: raw
    if LOG_SNAPSHOTS:
        _write_json(STATE_DIR / _snap_name("raw"), raw)

    # 2) filtrele
    try:
        filtered = filter_pipeline_v1(raw, hint=payload.hint)
    except Exception as e:
        raise HTTPException(500, f"filtering failed: {e}")

    # snapshot: filtered
    if LOG_SNAPSHOTS:
        _write_json(STATE_DIR / _snap_name("filtered"), filtered)

    return filtered

@app.post("/action")
def action_proxy(plan: Dict[str, Any]):
    """
    Guard → forward to SUT /action.
    Logs:
      - logs/action/plan_*.json           (LLM plan full JSON)
      - logs/action/result_*.json         (SUT dönüşü)
      - logs/reasoning/thoughts.log (append)  (plan.reasoning)
    """
    # 1) plan’ı kaydet (full)
    if LOG_SNAPSHOTS:
        _write_json(ACTION_DIR / _snap_name("plan"), plan)

    # 2) reasoning’i ayrı logla (human-friendly)
    if LOG_THOUGHTS:
        try:
            rid = plan.get("action_id") or "-"
            reason = plan.get("reasoning") or ""
            with (REASON_DIR / "thoughts.log").open("a", encoding="utf-8") as f:
                f.write(f"[{_ts()}] action_id={rid}\n")
                if reason:
                    f.write(reason.strip() + "\n")
                else:
                    f.write("(no reasoning)\n")
                f.write("-" * 60 + "\n")
        except Exception:
            pass

    # 3) guardrails
    try:
        safe_plan = guard_action_plan_v1(plan)
    except ValueError as ve:
        raise HTTPException(400, f"plan guard failed: {ve}")

    # 4) SUT’a ilet
    try:
        r = session.post(REAL_SUT_ACTION, json=safe_plan, timeout=20)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        raise HTTPException(502, f"SUT /action failed: {e}")

    # 5) SUT result snapshot
    if LOG_SNAPSHOTS:
        _write_json(ACTION_DIR / _snap_name("result"), body)

    return body
