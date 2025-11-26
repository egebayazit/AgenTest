# llm_service.py
# FastAPI service for LLM-based test automation - WITH OLLAMA SUPPORT

from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import logging
import os
import time
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# .env (optional)
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_service")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class StepIn(BaseModel):
    """Input model for a single test step"""
    test_step: str = Field(..., description="Manual test step description", min_length=1)
    expected_result: str = Field(..., description="What should be true after the step", min_length=1)
    note_to_llm: Optional[str] = Field(None, description="Optional hint for planner")

    @field_validator('test_step', 'expected_result')
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('must not be empty')
        return v.strip()


class LLMConfigOverride(BaseModel):
    """Runtime configuration overrides for LLM backend"""
    model: Optional[str] = Field(None, description="Override model name")
    max_tokens: Optional[int] = Field(None, ge=200, le=2000, description="Max tokens per response")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM temperature")
    post_action_delay: Optional[float] = Field(None, ge=0.0, le=3.0, description="Delay after action (seconds)")


class RunIn(BaseModel):
    """Input model for scenario run"""
    steps: List[StepIn] = Field(..., min_items=1, description="List of test steps to execute")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="LLM temperature (default: 0.1)")

    config: Optional[LLMConfigOverride] = Field(None, description="Optional backend config overrides")

    @field_validator('steps')
    def validate_steps(cls, v):
        if not v:
            raise ValueError('steps must contain at least one item')
        return v


class ConfigOut(BaseModel):
    """Configuration info response"""
    status: str
    provider: str
    state_urls: Dict[str, str]
    action_url: str
    model: Optional[str]
    json_mode: bool
    defaults: Dict[str, Any]


class HealthOut(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: float


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AgenTest LLM Service",
    description="LLM-powered UI test automation service"
)

# CORS for Streamlit/local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# BACKEND CREATION
# ============================================================================

def _mk_backend(
    model_override: Optional[str] = None,
    config: Optional[LLMConfigOverride] = None,
) -> LLMBackend:
    """
    Create LLM backend with optional configuration overrides.
    
    Required environment variables:
      - LLM_PROVIDER ("ollama","lmstudio", or "openrouter")
      - LLM_MODEL (model name)
      - LLM_BASE_URL (for Ollama, default: http://localhost:11434)
      - LLM_API_KEY (for OpenRouter only, optional for LM Studio)
      - SUT_STATE_URL_WINDRIVER
      - SUT_STATE_URL_ODS
      - SUT_ACTION_URL
    """
    # Get provider from env
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    # Model (override or env)
    model = model_override or os.getenv("LLM_MODEL", "mistral-small3.2:latest")
    
    # Base parameters
    params = {
        "llm_provider": provider,
        "llm_model": model,
    }
    
    # Provider-specific config
    if provider == "ollama":
        params["llm_base_url"] = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        params["llm_api_key"] = None  # Ollama doesn't need API key
        logger.info("Using Ollama provider: %s @ %s", model, params["llm_base_url"])
    
    elif provider == "lmstudio":
        params["llm_base_url"] = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        params["llm_api_key"] = os.getenv("LLM_API_KEY")  # Optional for LM Studio
        logger.info("Using LM Studio provider: %s @ %s", model, params["llm_base_url"])

    elif provider == "openrouter":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY required for OpenRouter provider")
        params["llm_api_key"] = api_key
        params["llm_base_url"] = "https://openrouter.ai/api/v1"
        logger.info("Using OpenRouter provider: %s", model)
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    
    # Apply config overrides
    if config:
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.post_action_delay is not None:
            params["post_action_delay"] = config.post_action_delay
    
    backend = LLMBackend.from_env(**params)
    return backend


# ============================================================================
# LOGGING HELPERS
# ============================================================================

def _print_action_details(index: int, action_log: ActionExecutionLog | Dict[str, Any]) -> None:
    """Print detailed action log"""
    is_dict = isinstance(action_log, dict)
    action_id = action_log["action_id"] if is_dict else action_log.action_id
    ack = action_log["ack"] if is_dict else action_log.ack
    plan = action_log["plan"] if is_dict else action_log.plan
    state_after = action_log.get("state_after") if is_dict else getattr(action_log, "state_after", None)

    logger.info("    Action #%d: %s", index, action_id)
    logger.info("      Status: %s", ack.get("status"))
    
    if ack.get("status") != "ok":
        logger.error("      Error: %s", ack.get("message", ""))

    reasoning = plan.get("reasoning")
    if reasoning:
        logger.info("      Reasoning: %s", reasoning[:150])

    steps = plan.get("steps", [])
    logger.info("      Steps: %d actions", len(steps))
    if steps and len(steps) <= 5:
        for step_idx, step in enumerate(steps, 1):
            step_type = step.get("type", "unknown")
            logger.info("        %02d: %s", step_idx, step_type)

    if state_after:
        element_count = len(state_after.get("elements", []))
        logger.info("      State after: %d elements", element_count)


def _print_step_outcome(index: int, outcome: ScenarioStepOutcome | Dict[str, Any]) -> None:
    """Print detailed step outcome"""
    is_dict = isinstance(outcome, dict)
    step = outcome["step"] if is_dict else outcome.step
    result = outcome["result"] if is_dict else outcome.result

    # Extract step fields
    if isinstance(step, dict):
        test_step = step.get("test_step")
        expected_result = step.get("expected_result")
        note_to_llm = step.get("note_to_llm")
    else:
        test_step = step.test_step
        expected_result = step.expected_result
        note_to_llm = step.note_to_llm

    # Extract result fields
    if isinstance(result, dict):
        res_status = result.get("status")
        res_reason = result.get("reason")
        res_attempts = result.get("attempts")
        actions = result.get("actions", [])
    else:
        res_status = result.status
        res_reason = result.reason
        res_attempts = result.attempts
        actions = result.actions

    logger.info("")
    logger.info("=" * 80)
    logger.info("Step %d: %s", index, test_step)
    logger.info("  Expected: %s", expected_result)
    if note_to_llm:
        logger.info("  Note to LLM: %s", note_to_llm)
    logger.info("  Result: %s (after %d attempts)", res_status, res_attempts)
    if res_reason:
        logger.info("  Reason: %s", res_reason)
    logger.info("=" * 80)

    for action_idx, action in enumerate(actions, 1):
        _print_action_details(action_idx, action)


def _make_debug_summary(result: ScenarioResult | Dict[str, Any]) -> Dict[str, Any]:
    """Create compact debug summary"""
    is_dict = isinstance(result, dict)
    status = result.get("status") if is_dict else result.status
    steps = result.get("steps", []) if is_dict else result.steps

    summary_steps: List[Dict[str, Any]] = []
    for i, outcome in enumerate(steps, 1):
        if isinstance(outcome, dict):
            step = outcome.get("step", {})
            res = outcome.get("result", {})
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
                "attempts": outcome.result.attempts,
            }
            actions = outcome.result.actions

        summary_steps.append({
            "index": i,
            "test_step": step.get("test_step", "")[:100],
            "expected_result": step.get("expected_result", "")[:100],
            "result_status": res.get("status"),
            "result_reason": res.get("reason", "")[:200] if res.get("reason") else None,
            "attempts": res.get("attempts"),
            "actions_count": len(actions or []),
        })

    return {
        "status": status,
        "total_steps": len(summary_steps),
        "passed_steps": sum(1 for s in summary_steps if s["result_status"] == "passed"),
        "failed_steps": sum(1 for s in summary_steps if s["result_status"] == "failed"),
        "error_steps": sum(1 for s in summary_steps if s["result_status"] == "error"),
        "steps": summary_steps,
    }


def _calculate_metrics(result: ScenarioResult, backend: LLMBackend, duration: float) -> Dict[str, Any]:
    """Calculate detailed metrics"""
    total_attempts = sum(s.result.attempts for s in result.steps)
    total_actions = sum(len(s.result.actions) for s in result.steps)
    
    return {
        "duration_sec": round(duration, 2),
        "total_steps": len(result.steps),
        "passed_steps": sum(1 for s in result.steps if s.result.status == "passed"),
        "failed_steps": sum(1 for s in result.steps if s.result.status == "failed"),
        "error_steps": sum(1 for s in result.steps if s.result.status == "error"),
        "total_attempts": total_attempts,
        "total_actions": total_actions,
        "avg_attempts_per_step": round(total_attempts / len(result.steps), 2) if result.steps else 0,
        "backend_config": {
            "provider": backend.llm_provider,
            "model": backend.model,
            "max_tokens": backend.max_tokens,
            "max_attempts": backend.max_attempts,
            "post_action_delay": backend.post_action_delay,
            "detection_strategy": "WinDriver (attempt 1) → ODS (attempt 2)",
        },
    }


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthOut)
async def root() -> HealthOut:
    """Root endpoint"""
    return HealthOut(
        status="ok",
        service="AgenTest LLM Service",
        timestamp=time.time(),
    )


@app.get("/healthz", response_model=HealthOut)
async def healthz() -> HealthOut:
    """Health check endpoint"""
    return HealthOut(
        status="ok",
        service="AgenTest LLM Service",
        timestamp=time.time(),
    )


@app.get("/config", response_model=ConfigOut)
async def get_config() -> ConfigOut:
    """Get current configuration"""
    state_url_windriver = os.getenv("SUT_STATE_URL_WINDRIVER", "http://127.0.0.1:18800/state/for-llm")
    state_url_ods = os.getenv("SUT_STATE_URL_ODS", "http://127.0.0.1:18800/state/from-ods")
    action_url = os.getenv("SUT_ACTION_URL", "http://192.168.137.249:18080/action")
    
    # LLM config
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "mistral-small3.2:latest")
    
    enforce = os.getenv("ENFORCE_JSON_RESPONSE", "1") not in ("0", "false", "False")
    
    return ConfigOut(
        status="ok",
        provider=provider,
        state_urls={
            "windriver": state_url_windriver,
            "ods": state_url_ods,
        },
        action_url=action_url,
        model=model,
        json_mode=enforce,
        defaults={
            "max_attempts": 2,
            "max_tokens": 600,
            "temperature": 0.1,
            "post_action_delay": 0.5,
            "schema_retry_limit": 1,
            "detection_strategy": "WinDriver (fast) → ODS (accurate)",
        },
    )


@app.post("/run")
async def run_scenario(body: RunIn) -> Dict[str, Any]:
    """
    Execute a test scenario with one or more steps.
        """
    # Provider check
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    # API key validation (only for OpenRouter)
    if provider == "openrouter":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="LLM_API_KEY missing in environment for OpenRouter provider."
            )
    elif provider not in ("ollama", "lmstudio", "openrouter"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LLM_PROVIDER: {provider}. Must be 'ollama', 'lmstudio', or 'openrouter'."
        )
    
    # Create backend with config overrides
    try:
        backend = _mk_backend(
            model_override=body.config.model if body.config else None,
            config=body.config,
        )
    except ValueError as e:
        logger.error("Backend configuration error: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend configuration: {str(e)}"
        )
    except Exception as e:
        logger.exception("Failed to create backend")
        raise HTTPException(
            status_code=500,
            detail=f"Backend initialization failed: {str(e)}"
        )
    
    # Convert to StepDefinition
    steps_def = [
        StepDefinition(
            test_step=s.test_step,
            expected_result=s.expected_result,
            note_to_llm=s.note_to_llm,
        )
        for s in body.steps
    ]
    
    # Log start
    logger.info("=" * 80)
    logger.info("SCENARIO START")
    logger.info("  Provider: %s", provider)
    logger.info("  Model: %s", backend.model)
    logger.info("  Steps: %d (each executed in ISOLATION)", len(steps_def))
    logger.info("  Detection strategy: WinDriver → ODS")
    logger.info("  Temperature: %.2f", body.temperature)
    logger.info("=" * 80)
    
    # Execute scenario
    started = time.time()
    try:
        result = await backend.run_scenario(
            steps=steps_def,
            temperature=body.temperature,
        )
    except BackendError as be:
        logger.exception("Backend error during scenario execution")
        raise HTTPException(
            status_code=502,
            detail=f"Backend error: {str(be)}"
        )
    except Exception as e:
        logger.exception("Unexpected error during scenario execution")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
    
    duration = time.time() - started
    
    # Log completion
    logger.info("=" * 80)
    logger.info("SCENARIO COMPLETE")
    logger.info("  Status: %s", result.status)
    logger.info("  Duration: %.2f seconds", duration)
    logger.info("  Steps executed: %d", len(result.steps))
    logger.info("=" * 80)
    
    # Print detailed step outcomes
    if result.steps:
        for i, outcome in enumerate(result.steps, 1):
            _print_step_outcome(i, outcome)
    
    # Build response
    payload = asdict(result)
    
    # Add metadata and metrics
    payload["_meta"] = _calculate_metrics(result, backend, duration)
    payload["_debug_summary"] = _make_debug_summary(result)
    
    return payload


# ============================================================================
# DEBUG ENDPOINTS
# ============================================================================

@app.post("/debug/expected-check")
async def debug_expected_check(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test the expected result checker.
    
    Useful for debugging why a step is/isn't passing.
    """
    state = body.get("state", {})
    expected = body.get("expected_result", "")
    
    if not expected:
        raise HTTPException(status_code=400, detail="expected_result required")
    
    try:
        backend = _mk_backend()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend init failed: {e}")
    
    holds = backend._expected_holds(state, expected)
    
    # Extract visible text for debugging
    visible_names = []
    for e in state.get("elements", []):
        name = e.get("name") or e.get("name_ocr", "")
        if name:
            visible_names.append(name)
    
    return {
        "status": "ok",
        "expected_result": expected,
        "holds": holds,
        "visible_elements_count": len(state.get("elements", [])),
        "visible_elements_sample": visible_names[:30],
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("LLM_SERVICE_PORT", "18888"))
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "mistral-small3.2:latest")
    
    logger.info("=" * 80)
    logger.info("Starting AgenTest LLM Service v3.0.0-ollama")
    logger.info("  Port: %d", port)
    logger.info("  Provider: %s", provider)
    logger.info("  Model: %s", model)
    
    if provider == "ollama":
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        logger.info("  Ollama URL: %s", base_url)

    elif provider == "lmstudio":
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        logger.info("  LM Studio URL: %s", base_url)
        api_key = os.getenv("LLM_API_KEY", "")
        logger.info("  API Key: %s", "✓ SET" if api_key else "✗ NOT SET (optional)")

    elif provider == "openrouter":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        logger.info("  API Key: %s", "✓ SET" if api_key else "✗ NOT SET")
    
    logger.info("  WinDriver URL: %s", os.getenv("SUT_STATE_URL_WINDRIVER", "NOT SET"))
    logger.info("  ODS URL: %s", os.getenv("SUT_STATE_URL_ODS", "NOT SET"))
    logger.info("  Action URL: %s", os.getenv("SUT_ACTION_URL", "NOT SET"))
    logger.info("  Detection strategy: WinDriver (fast) → ODS (accurate)")
    logger.info("=" * 80)
    
    uvicorn.run(
        "llm_service:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )