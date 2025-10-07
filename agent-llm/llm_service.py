from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import logging
import os
import time
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# .env (opsiyonel)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Backend
from llm_backend import (
    LLMBackend,
    StepDefinition,
    BackendError,
    ActionExecutionLog,
    ScenarioResult,
    ScenarioStepOutcome,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_service")

# ---------- Pydantic şemaları ----------

class StepIn(BaseModel):
    test_step: str = Field(..., description="Manual test step description")
    expected_result: str = Field(..., description="What should be true after the step")
    note_to_llm: Optional[str] = Field(None, description="Optional hint for planner")

class RunIn(BaseModel):
    steps: List[StepIn]
    temperature: float = 0.1
    max_attempts: Optional[int] = None
    # .env’yi override etmek istersen:
    model: Optional[str] = None

class ConfigOut(BaseModel):
    status: str
    state_url: str
    action_url: str
    model: Optional[str]
    json_mode: bool

# ---------- FastAPI app ----------

app = FastAPI(title="AgenTest LLM Service", version="0.1.0")

# Streamlit/yerel geliştirme için CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Yardımcılar (backend & logging) ----------

def _mk_backend(model_override: Optional[str] = None) -> LLMBackend:
    """
    Ortam değişkenlerinden LLMBackend kurar.
    Zorunlu env’ler:
      - OPENROUTER_API_KEY
      - OPENROUTER_MODEL (veya model_override)
      - SUT_STATE_URL
      - SUT_ACTION_URL
    """
    openrouter_model = model_override or os.getenv("OPENROUTER_MODEL", "")
    backend = LLMBackend.from_env(openrouter_model=openrouter_model)
    return backend


def _print_action_details(index: int, action_log: ActionExecutionLog | Dict[str, Any]) -> None:
    """Dataclass veya dict alır; her action için ayrıntılı log basar."""
    is_dict = isinstance(action_log, dict)
    action_id   = action_log["action_id"] if is_dict else action_log.action_id
    ack         = action_log["ack"]        if is_dict else action_log.ack
    plan        = action_log["plan"]       if is_dict else action_log.plan
    state_after = action_log.get("state_after") if is_dict else getattr(action_log, "state_after", None)

    logger.info("    Action #%d: %s", index, action_id)
    logger.info("      ACK: %s", json.dumps(ack, ensure_ascii=False))

    reasoning = plan.get("reasoning")
    if reasoning:
        logger.info("      Reasoning: %s", reasoning)

    steps = plan.get("steps", [])
    if steps:
        logger.info("      Steps:")
        for step_idx, step in enumerate(steps, 1):
            logger.info("        %02d: %s", step_idx, json.dumps(step, ensure_ascii=False))
    else:
        logger.info("      Steps: []")

    if state_after:
        hint = {
            "status": state_after.get("status"),
            "focused_element": state_after.get("focused_element"),
        }
        logger.info("      State after hint: %s", json.dumps(hint, ensure_ascii=False))


def _print_step_outcome(index: int, outcome: ScenarioStepOutcome | Dict[str, Any]) -> None:
    """Dataclass veya dict alır; bir senaryo adımını loglar."""
    is_dict = isinstance(outcome, dict)
    step    = outcome["step"]   if is_dict else outcome.step
    result  = outcome["result"] if is_dict else outcome.result

    # step alanları
    if isinstance(step, dict):
        test_step       = step.get("test_step")
        expected_result = step.get("expected_result")
        note_to_llm     = step.get("note_to_llm")
    else:
        test_step       = step.test_step
        expected_result = step.expected_result
        note_to_llm     = step.note_to_llm

    # result alanları
    if isinstance(result, dict):
        res_status = result.get("status")
        res_reason = result.get("reason")
        actions    = result.get("actions", [])
    else:
        res_status = result.status
        res_reason = result.reason
        actions    = result.actions

    logger.info("")
    logger.info("Step %d: %s", index, test_step)
    logger.info("  Expected: %s", expected_result)
    if note_to_llm:
        logger.info("  Note to LLM: %s", note_to_llm)
    logger.info("  Result: %s", res_status)
    if res_reason:
        logger.info("  Reason: %s", res_reason)

    for action_idx, action in enumerate(actions, 1):
        _print_action_details(action_idx, action)


def _make_debug_summary(result: ScenarioResult | Dict[str, Any]) -> Dict[str, Any]:
    """Response’a eklenecek hafif bir özet üretir."""
    is_dict = isinstance(result, dict)
    status  = result.get("status") if is_dict else result.status
    steps   = result.get("steps", []) if is_dict else result.steps

    summary_steps: List[Dict[str, Any]] = []
    for i, outcome in enumerate(steps, 1):
        if isinstance(outcome, dict):
            step = outcome.get("step", {})
            res  = outcome.get("result", {})
            actions = res.get("actions", [])
        else:
            step = {
                "test_step": outcome.step.test_step,
                "expected_result": outcome.step.expected_result,
                "note_to_llm": outcome.step.note_to_llm,
            }
            res = {
                "status": outcome.result.status,
                "reason": outcome.result.reason,
            }
            actions = outcome.result.actions

        summary_steps.append({
            "index": i,
            "test_step": step.get("test_step"),
            "expected_result": step.get("expected_result"),
            "result_status": res.get("status"),
            "result_reason": res.get("reason"),
            "actions_count": len(actions or []),
        })

    return {
        "status": status,
        "steps": summary_steps,
    }

# ---------- Endpoint’ler ----------

@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/config", response_model=ConfigOut)
async def get_config() -> ConfigOut:
    state_url = os.getenv("SUT_STATE_URL", "http://127.0.0.1:18080/state")
    action_url = os.getenv("SUT_ACTION_URL", "http://127.0.0.1:18080/action")
    model = os.getenv("OPENROUTER_MODEL")
    enforce = os.getenv("LLM_ENFORCE_JSON", "1") not in ("0", "false", "False")
    return ConfigOut(
        status="ok",
        state_url=state_url,
        action_url=action_url,
        model=model,
        json_mode=enforce,
    )

@app.post("/run")
async def run_scenario(body: RunIn) -> Dict[str, Any]:
    # Temel kontroller
    if not body.steps:
        raise HTTPException(status_code=400, detail="steps must not be empty")

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY missing in environment")

    # Backend’i hazırla (gerekirse model override)
    try:
        backend = _mk_backend(model_override=body.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to init backend: {e}")

    # StepDefinition listesine çevir
    steps_def = [
        StepDefinition(
            test_step=s.test_step,
            expected_result=s.expected_result,
            note_to_llm=s.note_to_llm,
        )
        for s in body.steps
    ]

    started = time.time()
    try:
        result = await backend.run_scenario(
            steps=steps_def,
            temperature=body.temperature,
            max_attempts=body.max_attempts,
        )
    except BackendError as be:
        logger.exception("Backend error during run")
        raise HTTPException(status_code=502, detail=str(be))
    except Exception as e:
        logger.exception("Unexpected error during run")
        raise HTTPException(status_code=500, detail=str(e))

    duration = time.time() - started

    # ---- Ayrıntılı loglar (konsola) ----
    logger.info("=== Scenario finished in %.3f sec, status=%s, model=%s ===",
                duration, result.status, backend.model)
    if result.steps:
        for i, outcome in enumerate(result.steps, 1):
            _print_step_outcome(i, outcome)
    else:
        logger.info("(No steps in result)")

    # dataclass -> dict
    payload = asdict(result)

    # Hızlı teşhis için küçük bir özet ekleyelim:
    payload["_meta"] = {
        "duration_sec": round(duration, 3),
        "attempts_total": sum((r.result.attempts for r in result.steps), 0) if result.steps else 0,
        "status": result.status,
        "model": backend.model,
    }
    payload["_debug_summary"] = _make_debug_summary(result)

    return payload


# Yerelden çalıştırmak için:
#   uvicorn llm_service:app --host 0.0.0.0 --port 18888 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_service:app",
                host="0.0.0.0",
                port=int(os.getenv("LLM_SERVICE_PORT", "18888")),
                reload=True)
