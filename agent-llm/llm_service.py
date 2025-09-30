# llm_service.py
# FastAPI wrapper around LLMBackend so that UI (or other clients) can call HTTP endpoints.
#
# Endpoints:
# - GET  /healthz         : service + config sanity
# - POST /state           : proxy to SUT /state   (optional)
# - POST /action          : proxy to SUT /action  (optional)
# - POST /plan_step       : produce an action plan (LLM only, no execution)
# - POST /run_step        : plan + execute + verify (single step)

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from llm_backend import LLMBackend, Config, TestStep

# Load .env safely (this folder, or parent)
HERE = Path(__file__).resolve().parent
for candidate in (HERE / ".env", HERE.parent / ".env"):
    if candidate.exists():
        load_dotenv(candidate)
        break
else:
    load_dotenv()

# ---------- I/O Schemas ----------

class StepIn(BaseModel):
    step: str
    expected_result: str = Field(..., alias="expected")
    note_to_llm: Optional[str] = Field("", alias="note")

class PlanOut(BaseModel):
    action_plan: Dict[str, Any]

class RunOut(BaseModel):
    success: bool
    passed: bool
    actual: str
    logs: List[str]
    action_plan: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None

# ---------- App + lazy backend ----------

app = FastAPI(title="Agentest LLM Backend", version="0.2.0")
_backend: Optional[LLMBackend] = None

def get_backend() -> LLMBackend:
    global _backend
    if _backend is None:
        cfg = Config(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            sut_state_url=os.getenv("SUT_STATE_URL", "http://127.0.0.1:18080/state"),
            sut_action_url=os.getenv("SUT_ACTION_URL", "http://127.0.0.1:18080/action"),
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
        )
        _backend = LLMBackend(cfg)
    return _backend

def ensure_key():
    if not os.getenv("OPENROUTER_API_KEY", ""):
        raise HTTPException(500, "OPENROUTER_API_KEY missing. Put it in .env or env vars.")

# ---------- Endpoints ----------

@app.get("/healthz")
def healthz():
    be = get_backend()
    return {
        "ok": True,
        "model": be.config.model,
        "has_api_key": bool(os.getenv("OPENROUTER_API_KEY", "")),
        "sut_state_url": be.config.sut_state_url,
        "sut_action_url": be.config.sut_action_url,
    }

# optional proxies
@app.post("/state")
def proxy_state():
    be = get_backend()
    s = be.get_state_from_sut()
    if not s:
        raise HTTPException(502, "SUT /state failed")
    return s

@app.post("/action")
def proxy_action(body: Dict[str, Any]):
    be = get_backend()
    ok = be.execute_action(body)
    if not ok:
        raise HTTPException(502, "SUT /action failed")
    return {"status": "ok"}

@app.post("/plan_step", response_model=PlanOut)
def plan_step(req: StepIn):
    ensure_key()
    be = get_backend()
    state = be.get_state_from_sut()
    if not state:
        raise HTTPException(502, "SUT /state failed")
    plan = be.call_llm(TestStep(req.step, req.expected_result, req.note_to_llm), state)
    if not plan:
        raise HTTPException(500, "LLM plan generation failed")
    return {"action_plan": plan}

@app.post("/run_step", response_model=RunOut)
def run_step(req: StepIn):
    ensure_key()
    be = get_backend()
    return be.run_test_step(TestStep(req.step, req.expected_result, req.note_to_llm))
# ---------- Run with: uvicorn llm_service:app --reload --port 18081 ----------