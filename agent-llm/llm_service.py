# llm_service.py v3.0.1
# FastAPI service for LLM-based test automation - WITH OLLAMA, LM STUDIO, OPENROUTER, AND LLAMA.CPP SUPPORT

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
import httpx

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
    RawODSBackend,
    BackendError,
    ActionExecutionLog,
    ScenarioResult,
    ScenarioStepOutcome,
    LLAMACPP_AVAILABLE,
    ANTHROPIC_AVAILABLE,
    GEMINI_AVAILABLE,
)

# Agent Mode Backend
from agent_mode_backend import (
    AgentModeBackend,
    AgentInstruction,
    AgentStepResult,
    AgentScenarioResult,
    AgentScenarioOutcome,
    AgentActionLog,
)

# Settings Manager
from settings_manager import get_settings_manager

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
    scenario_name: str = Field("Untitled_Test", description="Name of the test scenario (used for saving record)")
    steps: List[StepIn] = Field(..., min_items=1, description="List of test steps to execute")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="LLM temperature (default: 0.1)")

    config: Optional[LLMConfigOverride] = Field(None, description="Optional backend config overrides")

    @field_validator('steps')
    def validate_steps(cls, v):
        if not v:
            raise ValueError('steps must contain at least one item')
        return v
    
class ReplayIn(BaseModel):
    """Input model for replaying a saved test"""
    test_name: str = Field(..., description="Name of the saved test (e.g., 'Login_Test_01')")


class ConfigOut(BaseModel):
    """Configuration info response"""
    status: str
    provider: str
    state_urls: Dict[str, str]
    action_url: str
    model: Optional[str]
    json_mode: bool
    llamacpp_available: bool
    defaults: Dict[str, Any]


class HealthOut(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: float


class LLMSettingsIn(BaseModel):
    """Input model for LLM settings"""
    provider: str = Field(..., description="LLM provider (ollama, openrouter, anthropic, gemini, custom)")
    base_url: str = Field("", description="Base URL for the LLM API")
    api_key: Optional[str] = Field(None, description="API key (will be encrypted)")
    model: str = Field(..., description="Model name")


# ============================================================================
# AGENT MODE PYDANTIC MODELS
# ============================================================================

class AgentStepIn(BaseModel):
    """Input model for a single agent instruction - no expected result"""
    instruction: str = Field(..., description="What the agent should do", min_length=1)
    note: Optional[str] = Field(None, description="Additional notes/hints for the agent")

    @field_validator('instruction')
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('must not be empty')
        return v.strip()


class AgentRunIn(BaseModel):
    """Input model for agent mode scenario run"""
    scenario_name: str = Field("Untitled_Agent_Task", description="Name of the agent task")
    instructions: List[AgentStepIn] = Field(..., min_items=1, description="List of instructions to execute")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="LLM temperature (default: 0.1)")
    use_ods: bool = Field(False, description="Use ODS instead of WinDriver for element detection")

    @field_validator('instructions')
    def validate_instructions(cls, v):
        if not v:
            raise ValueError('instructions must contain at least one item')
        return v


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AgenTest LLM Service",
    description="LLM-powered UI test automation service",
    version="3.0.1"
)

# CORS for Streamlit/local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom logging handler to capture logs for UI
ui_logs: List[str] = []

# Global execution control
execution_cancelled = False

class UILogHandler(logging.Handler):
    """Captures logs with emojis for UI display."""
    
    # Only capture logs that start with these prefixes (simplified UI logs)
    UI_PREFIXES = ('STARTING', 'STEP', 'Attempt', 'PRE-CHECK', 'SENDING', 'PARSED', 'EXECUTING', 'VALIDATION', 'PASSED', 'FAILED', 'STOPPED', 'Planning failed', 'Action failed', 'TOKEN USAGE')
    
    def emit(self, record):
        msg = self.format(record)
        # Only add to UI logs if it contains a relevant prefix
        for prefix in self.UI_PREFIXES:
            if prefix in msg:
                ui_logs.append(msg)
                break

# Add UI handler to llm_backend logger
ui_handler = UILogHandler()
ui_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger("llm_backend").addHandler(ui_handler)

# Add UI handler to agent_mode_backend logger (for Agent Mode)
logging.getLogger("agent_mode_backend").addHandler(ui_handler)

# ============================================================================
# BACKEND CREATION
# ============================================================================

def _mk_backend(
    model_override: Optional[str] = None,
    config: Optional[LLMConfigOverride] = None,
) -> LLMBackend:
    """
    Create LLM backend with optional configuration overrides.
    
    Configuration priority:
      1. Runtime overrides (model_override, config)
      2. Settings from ~/.agentest/llm_settings.json
      3. Environment variables (.env)
      
    Required environment variables (if settings not configured):
      - LLM_PROVIDER ("ollama", "lmstudio", "openrouter", "custom", or "llamacpp")
      - LLM_MODEL (model name or path to .gguf file for llamacpp)
      - LLM_BASE_URL (for Ollama, default: http://localhost:11434)
      - LLM_API_KEY (for OpenRouter only, optional for LM Studio)
      - SUT_STATE_URL_WINDRIVER
      - SUT_STATE_URL_ODS
      - SUT_ACTION_URL
    """
    # Try to get settings from settings manager first
    settings_manager = get_settings_manager()
    saved_settings = settings_manager.get_settings(include_raw_key=True)
    
    # Priority: override > settings > env
    provider = saved_settings.get("provider") or os.getenv("LLM_PROVIDER", "ollama")
    model = model_override or saved_settings.get("model") or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q6_k")
    base_url = saved_settings.get("base_url") or ""
    api_key = saved_settings.get("api_key") or ""
    
    # Base parameters
    params = {
        "llm_provider": provider,
        "llm_model": model,
    }
    
    # Provider-specific config
    if provider == "local":
        # Local provider: uses Ollama API with fixed Qwen model and LOCAL_SYSTEM_PROMPT
        params["llm_base_url"] = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        params["llm_api_key"] = None  # Local doesn't need API key
        params["llm_model"] = "qwen2.5:7b-instruct-q6_k"  # Fixed model for local
        logger.info("Using Local provider (Qwen): %s @ %s", params["llm_model"], params["llm_base_url"])
    
    elif provider == "ollama":
        params["llm_base_url"] = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        params["llm_api_key"] = None  # Ollama doesn't need API key
        logger.info("Using Ollama provider: %s @ %s", model, params["llm_base_url"])
    
    elif provider == "lmstudio":
        params["llm_base_url"] = base_url or os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        params["llm_api_key"] = api_key or os.getenv("LLM_API_KEY")  # Optional for LM Studio
        logger.info("Using LM Studio provider: %s @ %s", model, params["llm_base_url"])

    elif provider == "openrouter":
        final_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not final_key:
            raise ValueError("LLM_API_KEY required for OpenRouter provider")
        params["llm_api_key"] = final_key
        params["llm_base_url"] = "https://openrouter.ai/api/v1"
        logger.info("Using OpenRouter provider: %s", model)
    
    elif provider == "llamacpp":
        if not LLAMACPP_AVAILABLE:
            raise ValueError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

        # For llamacpp, model should be path to .gguf file
        params["llm_base_url"] = ""  # Not used for llamacpp
        params["llm_api_key"] = None  # Not used for llamacpp

        # Optional llamacpp parameters
        params["llamacpp_n_ctx"] = int(os.getenv("LLAMACPP_N_CTX", "4096"))
        params["llamacpp_n_gpu_layers"] = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "0"))

        logger.info("Using llama.cpp provider: %s", model)
        logger.info("  Context window: %d", params["llamacpp_n_ctx"])
        logger.info("  GPU layers: %d", params["llamacpp_n_gpu_layers"])

    elif provider == "openai":
        params["llm_base_url"] = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8090/v1")
        params["llm_api_key"] = api_key or os.getenv("LLM_API_KEY")  # Optional (dummy for llama-server)
        logger.info("Using OpenAI-compatible provider: %s @ %s", model, params["llm_base_url"])

    elif provider == "anthropic":
        final_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not final_key:
            raise ValueError("LLM_API_KEY required for Anthropic provider")
        params["llm_api_key"] = final_key
        params["llm_base_url"] = ""  # Not used for Anthropic
        logger.info("Using Anthropic provider: %s", model)

    elif provider == "gemini":
        final_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not final_key:
            raise ValueError("LLM_API_KEY required for Gemini provider")
        params["llm_api_key"] = final_key
        params["llm_base_url"] = ""  # Not used for Gemini
        logger.info("Using Gemini provider: %s", model)

    elif provider == "custom":
        # Custom OpenAI-compatible provider
        if not base_url:
            raise ValueError("Base URL required for Custom provider. Set it in Settings or LLM_BASE_URL env var.")
        params["llm_base_url"] = base_url
        params["llm_api_key"] = api_key or os.getenv("LLM_API_KEY")
        logger.info("Using Custom (OpenAI-compatible) provider: %s @ %s", model, params["llm_base_url"])

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

def _mk_raw_backend(
    model_override: Optional[str] = None,
    config: Optional[LLMConfigOverride] = None,
) -> RawODSBackend:
    """
    Creates a RawODSBackend instance (No filters, No WinDriver, No retries).
    Used for benchmarking and demonstration.
    """
    # Mevcut _mk_backend mantığının aynısını kullanıyoruz, sadece sınıf farklı
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = model_override or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q6_k")
    
    params = {
        "llm_provider": provider,
        "llm_model": model,
    }
    
    # Provider ayarları (Mevcut koddan kopyalandı)
    if provider == "ollama":
        params["llm_base_url"] = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        params["llm_api_key"] = None
    elif provider == "lmstudio":
        params["llm_base_url"] = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        params["llm_api_key"] = os.getenv("LLM_API_KEY")
    elif provider == "openrouter":
        params["llm_api_key"] = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        params["llm_base_url"] = "https://openrouter.ai/api/v1"
    elif provider == "llamacpp":
        params["llm_base_url"] = ""
        params["llm_api_key"] = None
        params["llamacpp_n_ctx"] = int(os.getenv("LLAMACPP_N_CTX", "4096"))
        params["llamacpp_n_gpu_layers"] = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "0"))
    elif provider == "openai":
        params["llm_base_url"] = os.getenv("LLM_BASE_URL", "http://localhost:8090/v1")
        params["llm_api_key"] = os.getenv("LLM_API_KEY")
    elif provider == "gemini":
        params["llm_api_key"] = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        params["llm_base_url"] = ""
    elif provider == "anthropic":
        params["llm_api_key"] = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        params["llm_base_url"] = ""

    # Config override
    if config:
        if config.max_tokens is not None: params["max_tokens"] = config.max_tokens
        if config.post_action_delay is not None: params["post_action_delay"] = config.post_action_delay
    
    # RawODSBackend döndür
    return RawODSBackend.from_env(**params)


def _mk_agent_backend(
    model_override: Optional[str] = None,
) -> AgentModeBackend:
    """
    Creates an AgentModeBackend instance for Agent Mode.
    No semantic filtering, no pre-check, no expected result validation.
    """
    # Try to get settings from settings manager first
    settings_manager = get_settings_manager()
    saved_settings = settings_manager.get_settings(include_raw_key=True)
    
    # Priority: override > settings > env
    provider = saved_settings.get("provider") or os.getenv("LLM_PROVIDER", "ollama")
    model = model_override or saved_settings.get("model") or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q6_k")
    base_url = saved_settings.get("base_url") or ""
    api_key = saved_settings.get("api_key") or ""
    
    params = {
        "llm_provider": provider,
        "llm_model": model,
    }
    
    # Provider-specific config (same as _mk_backend)
    if provider == "local":
        params["llm_base_url"] = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        params["llm_api_key"] = None
        params["llm_model"] = "qwen2.5:7b-instruct-q6_k"
        logger.info("Agent Mode: Using Local provider (Qwen): %s", params["llm_model"])
    
    elif provider == "ollama":
        params["llm_base_url"] = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        params["llm_api_key"] = None
        logger.info("Agent Mode: Using Ollama provider: %s", model)
    
    elif provider == "lmstudio":
        params["llm_base_url"] = base_url or os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        params["llm_api_key"] = api_key or os.getenv("LLM_API_KEY")
        logger.info("Agent Mode: Using LM Studio provider: %s", model)

    elif provider == "openrouter":
        final_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not final_key:
            raise ValueError("LLM_API_KEY required for OpenRouter provider")
        params["llm_api_key"] = final_key
        params["llm_base_url"] = "https://openrouter.ai/api/v1"
        logger.info("Agent Mode: Using OpenRouter provider: %s", model)
    
    elif provider == "anthropic":
        final_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not final_key:
            raise ValueError("LLM_API_KEY required for Anthropic provider")
        params["llm_api_key"] = final_key
        params["llm_base_url"] = ""
        logger.info("Agent Mode: Using Anthropic provider: %s", model)

    elif provider == "gemini":
        final_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not final_key:
            raise ValueError("LLM_API_KEY required for Gemini provider")
        params["llm_api_key"] = final_key
        params["llm_base_url"] = ""
        logger.info("Agent Mode: Using Gemini provider: %s", model)

    elif provider == "openai":
        params["llm_base_url"] = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8090/v1")
        params["llm_api_key"] = api_key or os.getenv("LLM_API_KEY")
        logger.info("Agent Mode: Using OpenAI-compatible provider: %s", model)

    elif provider == "custom":
        if not base_url:
            raise ValueError("Base URL required for Custom provider.")
        params["llm_base_url"] = base_url
        params["llm_api_key"] = api_key or os.getenv("LLM_API_KEY")
        logger.info("Agent Mode: Using Custom provider: %s", model)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    
    return AgentModeBackend.from_env(**params)


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
    
    metrics = {
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
    
    # Add llamacpp-specific config if applicable
    if backend.llm_provider == "llamacpp":
        metrics["backend_config"]["llamacpp_n_ctx"] = backend.llamacpp_n_ctx
        metrics["backend_config"]["llamacpp_n_gpu_layers"] = backend.llamacpp_n_gpu_layers
    
    return metrics


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthOut)
async def root() -> HealthOut:
    """Root endpoint"""
    return HealthOut(
        status="ok",
        service="AgenTest LLM Service",
        version="3.0.1",
        timestamp=time.time(),
    )


@app.get("/healthz", response_model=HealthOut)
async def healthz() -> HealthOut:
    """Health check endpoint"""
    return HealthOut(
        status="ok",
        service="AgenTest LLM Service",
        version="3.0.1",
        timestamp=time.time(),
    )


@app.get("/logs")
async def get_logs() -> Dict[str, Any]:
    """Get current execution logs for UI polling"""
    return {"logs": ui_logs}


@app.post("/stop")
async def stop_execution() -> Dict[str, Any]:
    """Stop the currently running test execution"""
    global execution_cancelled
    execution_cancelled = True
    logger.info("Stop requested by user")
    return {"status": "ok", "message": "Stop requested"}


# ============================================================================
# SETTINGS ENDPOINTS
# ============================================================================

@app.get("/settings")
async def get_settings() -> Dict[str, Any]:
    """Get current LLM settings (API keys are masked)"""
    settings_manager = get_settings_manager()
    settings = settings_manager.get_settings(include_raw_key=False)
    return {
        "status": "ok",
        "settings": settings
    }


@app.post("/settings")
async def save_settings(body: LLMSettingsIn) -> Dict[str, Any]:
    """Save LLM settings (API key will be encrypted)"""
    settings_manager = get_settings_manager()
    success = settings_manager.save_settings(body.model_dump())
    if success:
        logger.info("LLM settings updated: provider=%s, model=%s", body.provider, body.model)
        return {"status": "ok", "message": "Settings saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")


@app.post("/settings/test")
async def test_llm_connection() -> Dict[str, Any]:
    """Test LLM connection with current settings"""
    settings_manager = get_settings_manager()
    success, message = settings_manager.test_connection()
    return {
        "status": "ok" if success else "error",
        "success": success,
        "message": message
    }


@app.get("/settings/providers")
async def get_providers() -> Dict[str, Any]:
    """Get list of available LLM providers"""
    settings_manager = get_settings_manager()
    return {
        "status": "ok",
        "providers": settings_manager.get_providers()
    }


@app.get("/config", response_model=ConfigOut)
async def get_config() -> ConfigOut:
    """Get current configuration"""
    state_url_windriver = os.getenv("SUT_STATE_URL_WINDRIVER", "http://127.0.0.1:18800/state/for-llm")
    state_url_ods = os.getenv("SUT_STATE_URL_ODS", "http://127.0.0.1:18800/state/from-ods")
    action_url = os.getenv("SUT_ACTION_URL", "http://192.168.137.52:18080/action")
    
    # LLM config - read from settings first, then env
    settings_manager = get_settings_manager()
    saved_settings = settings_manager.get_settings(include_raw_key=False)
    
    provider = saved_settings.get("provider") or os.getenv("LLM_PROVIDER", "ollama")
    model = saved_settings.get("model") or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q6_k")
    
    enforce = os.getenv("ENFORCE_JSON_RESPONSE", "1") not in ("0", "false", "False")
    
    defaults = {
        "max_attempts": 2,
        "max_tokens": 600,
        "temperature": 0.1,
        "post_action_delay": 0.5,
        "schema_retry_limit": 1,
        "detection_strategy": "WinDriver (fast) → ODS (accurate)",
    }
    
    # Add llamacpp-specific defaults if provider is llamacpp
    if provider == "llamacpp":
        defaults["llamacpp_n_ctx"] = int(os.getenv("LLAMACPP_N_CTX", "4096"))
        defaults["llamacpp_n_gpu_layers"] = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "0"))
    
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
        llamacpp_available=LLAMACPP_AVAILABLE,
        defaults=defaults,
    )

@app.post("/run")
async def run_scenario(body: RunIn) -> Dict[str, Any]:
    """
    Execute a test scenario with one or more steps.
    """
    # Provider check
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    # Provider validation
    if provider not in ("local", "ollama", "lmstudio", "openrouter", "llamacpp", "openai", "anthropic", "gemini", "custom"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LLM_PROVIDER: {provider}. Must be 'local', 'ollama', 'lmstudio', 'openrouter', 'llamacpp', 'openai', 'anthropic', 'gemini', or 'custom'."
        )
        
    # API key validation (for OpenRouter and Anthropic)
    if provider == "openrouter":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="LLM_API_KEY missing in environment for OpenRouter provider."
            )
    
    if provider == "anthropic":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="LLM_API_KEY missing in environment for Anthropic provider."
            )
        if not ANTHROPIC_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="anthropic package not installed. Install with: pip install anthropic"
            )
    
    if provider == "gemini":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="LLM_API_KEY missing in environment for Gemini provider."
            )
        if not GEMINI_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="google-genai package not installed. Install with: pip install google-genai"
            )
    
    # llamacpp availability check
    if provider == "llamacpp" and not LLAMACPP_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="llama-cpp-python not installed. Install with: pip install llama-cpp-python"
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
    
    # Clear previous UI logs
    ui_logs.clear()
    
    # Reset cancellation flag
    global execution_cancelled
    execution_cancelled = False
    
    # Cancel check callback
    def check_cancelled():
        return execution_cancelled
    
    # Execute scenario (logs captured automatically by UILogHandler)
    started = time.time()
    try:
        result = await backend.run_scenario(
            scenario_name=body.scenario_name,
            steps=steps_def,
            temperature=body.temperature,
            save_recording=True,
            cancel_check=check_cancelled
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

@app.post("/run-raw")
async def run_scenario_raw(body: RunIn) -> Dict[str, Any]:
    """
    DEMO ENDPOINT: Execute scenario using RAW ODS (No optimizations).
    - No WinDriver Fallback
    - No Semantic Filtering (Sends ALL elements)
    - No Pre-checks
    - No Retries
    """
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    # Backend oluştur (Raw versiyon)
    try:
        backend = _mk_raw_backend(
            model_override=body.config.model if body.config else None,
            config=body.config,
        )
    except Exception as e:
        logger.exception("Failed to create RAW backend")
        raise HTTPException(status_code=500, detail=f"Backend init failed: {str(e)}")
    
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
    logger.info("SCENARIO START -- RAW ODS MODE (UNOPTIMIZED)")
    logger.info("   WARNING: This mode is for demonstration only.")
    logger.info("   It sends unfiltered ODS data directly to LLM.")
    logger.info("=" * 80)
    
    started = time.time()
    try:
        result = await backend.run_scenario(
            steps=steps_def,
            temperature=body.temperature,
        )
    except Exception as e:
        logger.exception("Error during RAW scenario execution")
        raise HTTPException(status_code=500, detail=str(e))
    
    duration = time.time() - started
    
    # Log completion
    logger.info("=" * 80)
    logger.info("RAW SCENARIO COMPLETE")
    logger.info("  Status: %s", result.status)
    logger.info("  Duration: %.2f seconds", duration)
    logger.info("=" * 80)
    
    if result.steps:
        for i, outcome in enumerate(result.steps, 1):
            _print_step_outcome(i, outcome)
    
    payload = asdict(result)
    
    # Metrics'e RAW notu düşelim
    metrics = _calculate_metrics(result, backend, duration)
    metrics["backend_config"]["detection_strategy"] = "RAW ODS (Unfiltered, Single Shot)"
    metrics["mode"] = "demonstration_unoptimized"
    
    payload["_meta"] = metrics
    payload["_debug_summary"] = _make_debug_summary(result)
    
    return payload


# ============================================================================
# AGENT MODE ENDPOINT
# ============================================================================

@app.post("/run-agent")
async def run_agent_scenario(body: AgentRunIn) -> Dict[str, Any]:
    """
    Execute instructions in Agent Mode.
    
    Agent Mode is a simplified execution mode:
    - No semantic filtering
    - No pre-check
    - No expected result validation
    - Just execute instructions and report
    """
    # Get current settings for provider info
    settings_manager = get_settings_manager()
    saved_settings = settings_manager.get_settings(include_raw_key=True)
    provider = saved_settings.get("provider") or os.getenv("LLM_PROVIDER", "ollama")
    
    # Provider validation
    if provider not in ("local", "ollama", "lmstudio", "openrouter", "anthropic", "gemini", "openai", "custom"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LLM_PROVIDER: {provider}. Must be 'local', 'ollama', 'lmstudio', 'openrouter', 'anthropic', 'gemini', 'openai', or 'custom'."
        )
    
    # API key validation for providers that require it
    if provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="anthropic package not installed. Install with: pip install anthropic"
            )
    
    if provider == "gemini":
        if not GEMINI_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="google-genai package not installed. Install with: pip install google-genai"
            )
    
    # Create agent backend
    try:
        backend = _mk_agent_backend()
    except ValueError as e:
        logger.error("Agent backend configuration error: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend configuration: {str(e)}"
        )
    except Exception as e:
        logger.exception("Failed to create agent backend")
        raise HTTPException(
            status_code=500,
            detail=f"Agent backend initialization failed: {str(e)}"
        )
    
    # Convert to AgentInstruction
    instructions = [
        AgentInstruction(
            instruction=s.instruction,
            note=s.note,
        )
        for s in body.instructions
    ]
    
    # Clear previous UI logs
    ui_logs.clear()
    
    # Reset cancellation flag
    global execution_cancelled
    execution_cancelled = False
    
    # Cancel check callback
    def check_cancelled():
        return execution_cancelled
    
    # Log start
    logger.info("=" * 80)
    logger.info("AGENT MODE SCENARIO START: %s", body.scenario_name)
    logger.info("  Instructions: %d", len(instructions))
    logger.info("  Use ODS: %s", body.use_ods)
    logger.info("  Provider: %s", provider)
    logger.info("=" * 80)
    
    # Execute scenario
    started = time.time()
    try:
        result = await backend.run_scenario(
            scenario_name=body.scenario_name,
            instructions=instructions,
            temperature=body.temperature,
            use_ods=body.use_ods,
            cancel_check=check_cancelled
        )
    except Exception as e:
        logger.exception("Agent mode execution error")
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution error: {str(e)}"
        )
    
    duration = time.time() - started
    
    # Log completion
    logger.info("=" * 80)
    logger.info("AGENT MODE SCENARIO COMPLETE")
    logger.info("  Status: %s", result.status)
    logger.info("  Duration: %.2f seconds", duration)
    logger.info("  Instructions executed: %d", len(result.steps))
    logger.info("=" * 80)
    
    # Build response - convert dataclass to dict
    def action_to_dict(action):
        return {
            "action_id": action.action_id,
            "plan": action.plan,
            "ack": action.ack,
            "state_before": action.state_before,
            "state_after": action.state_after,
            "timestamp": action.timestamp,
            "token_usage": action.token_usage,
        }
    
    def step_result_to_dict(step_result):
        return {
            "status": step_result.status,
            "actions": [action_to_dict(a) for a in step_result.actions],
            "final_state": step_result.final_state,
            "last_plan": step_result.last_plan,
            "reason": step_result.reason,
        }
    
    def outcome_to_dict(outcome):
        return {
            "instruction": {
                "instruction": outcome.instruction.instruction,
                "note": outcome.instruction.note,
            },
            "result": step_result_to_dict(outcome.result),
        }
    
    payload = {
        "status": result.status,
        "steps": [outcome_to_dict(o) for o in result.steps],
        "final_state": result.final_state,
        "reason": result.reason,
        "_meta": {
            "mode": "agent",
            "duration_sec": round(duration, 2),
            "total_instructions": len(result.steps),
            "executed_count": sum(1 for s in result.steps if s.result.status == "executed"),
            "failed_count": sum(1 for s in result.steps if s.result.status == "failed"),
            "provider": provider,
            "use_ods": body.use_ods,
        }
    }
    
    # Save agent test to saved_tests directory
    try:
        import re
        safe_name = re.sub(r'[^\w\-_\. ]', '', body.scenario_name).replace(' ', '_')
        filename = f"{safe_name}.json"
        directory = "saved_tests"
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, filename)
        
        # Build merged steps from all actions for replay
        merged_steps = []
        for outcome in result.steps:
            for action in outcome.result.actions:
                if action.plan and action.plan.get("steps"):
                    merged_steps.extend(action.plan["steps"])
                    # Add wait between instructions for replay safety
                    merged_steps.append({"type": "wait", "ms": 1000})
        
        # Remove trailing wait
        if merged_steps and merged_steps[-1].get("type") == "wait":
            merged_steps.pop()
        
        # Build step definitions for Agent Mode (instruction-based)
        step_definitions = [
            {
                "test_step": s.instruction,
                "expected_result": "(Agent Mode - no expected result)",
                "note_to_llm": s.note
            }
            for s in body.instructions
        ]
        
        # Final payload for saving
        save_payload = {
            "action_id": f"replay_{safe_name}_{int(time.time())}",
            "coords_space": "physical",
            "steps": merged_steps,
            "step_definitions": step_definitions,
            "execution_result": payload["steps"],
            "execution_duration": duration,
            "mode": "agent",  # Mark as agent mode test
            "use_ods": body.use_ods
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_payload, f, indent=2, ensure_ascii=False)
        
        logger.info("AGENT TEST SAVED: %s", filepath)
    except Exception as e:
        logger.error("Failed to save agent test: %s", e)
    
    return payload


@app.post("/run-saved-test")
async def run_saved_test(body: ReplayIn) -> Dict[str, Any]:
    """
    Replay a previously recorded test scenario directly to the Action Endpoint.
    Bypasses LLM/Vision logic for maximum speed.
    """
    import re
    
    # 1. Dosya Yolunu Bul
    # İsimdeki boşlukları temizle (kaydederken yaptığımız gibi)
    safe_name = re.sub(r'[^\w\-_\. ]', '', body.test_name).replace(' ', '_')
    if not safe_name.endswith(".json"):
        safe_name += ".json"
        
    filepath = os.path.join("saved_tests", safe_name)
    
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404, 
            detail=f"Saved test not found: {safe_name}. (Looked in ./saved_tests/)"
        )

    # 2. Dosyayı Oku
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as e:
        logger.error("Failed to read test file: %s", e)
        raise HTTPException(status_code=500, detail=f"Corrupted test file: {str(e)}")

    # 3. Payload Kontrolü (Basitçe 'steps' var mı diye bakıyoruz)
    if "steps" not in payload:
        raise HTTPException(status_code=400, detail="Invalid test file format: missing 'steps'")

    # 4. Action Endpoint'e Gönder
    action_url = os.getenv("SUT_ACTION_URL", "http://192.168.137.52:18080/action")
    
    logger.info("=" * 80)
    logger.info("↺ REPLAYING SAVED TEST: %s", safe_name)
    logger.info("  Target URL: %s", action_url)
    logger.info("  Steps: %d", len(payload["steps"]))
    logger.info("=" * 80)

    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Action endpoint'e SADECE gerekli field'ları gönder
            # step_definitions metadata olarak kalıyor, action'a gitmiyor
            action_payload = {
                "action_id": payload.get("action_id", f"replay_{int(time.time())}"),
                "coords_space": payload.get("coords_space", "physical"),
                "steps": payload["steps"]
            }
            resp = await client.post(action_url, json=action_payload)
            resp.raise_for_status()
            sut_response = resp.json()
            
    except httpx.HTTPError as e:
        logger.error("SUT Communication Error: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to communicate with Action Endpoint: {str(e)}")
    except Exception as e:
        logger.error("Replay Error: %s", e)
        raise HTTPException(status_code=500, detail=f"Replay failed: {str(e)}")

    duration = time.time() - started
    
    # 5. Sonucu Dön
    return {
        "status": "success",
        "test_name": safe_name,
        "replay_duration_sec": round(duration, 2),
        "sut_response": sut_response
    }


# ============================================================================
# SCENARIO MANAGEMENT ENDPOINTS (for UI)
# ============================================================================

@app.get("/list-tests")
async def list_saved_tests() -> Dict[str, Any]:
    """
    List all saved test scenarios from the saved_tests directory.
    Returns basic metadata for each test.
    """
    import glob
    
    directory = "saved_tests"
    tests = []
    
    if not os.path.exists(directory):
        return {"status": "ok", "tests": [], "count": 0}
    
    for filepath in glob.glob(os.path.join(directory, "*.json")):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = os.path.basename(filepath)
            name = filename.replace(".json", "")
            
            tests.append({
                "name": name,
                "filename": filename,
                "steps_count": len(data.get("steps", [])),
                "action_id": data.get("action_id", ""),
                "modified_at": os.path.getmtime(filepath),
                "mode": data.get("mode", "test")  # "test" or "agent"
            })
        except Exception as e:
            logger.warning("Failed to read test file %s: %s", filepath, e)
            continue
    
    # Sort by modification time (newest first)
    tests.sort(key=lambda x: x["modified_at"], reverse=True)
    
    return {
        "status": "ok",
        "tests": tests,
        "count": len(tests)
    }


@app.delete("/delete-test/{test_name}")
async def delete_saved_test(test_name: str) -> Dict[str, Any]:
    """
    Delete a saved test scenario.
    """
    import re
    
    safe_name = re.sub(r'[^\w\-_\. ]', '', test_name).replace(' ', '_')
    if not safe_name.endswith(".json"):
        safe_name += ".json"
    
    filepath = os.path.join("saved_tests", safe_name)
    
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"Test not found: {safe_name}"
        )
    
    try:
        os.remove(filepath)
        logger.info("Deleted test: %s", safe_name)
        return {"status": "ok", "deleted": safe_name}
    except Exception as e:
        logger.error("Failed to delete test: %s", e)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/get-test/{test_name}")
async def get_test_details(test_name: str) -> Dict[str, Any]:
    """
    Get full details of a saved test scenario.
    """
    import re
    
    safe_name = re.sub(r'[^\w\-_\. ]', '', test_name).replace(' ', '_')
    if not safe_name.endswith(".json"):
        safe_name += ".json"
    
    filepath = os.path.join("saved_tests", safe_name)
    
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"Test not found: {safe_name}"
        )
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "status": "ok",
            "name": test_name,
            "filename": safe_name,
            "data": data
        }
    except Exception as e:
        logger.error("Failed to read test: %s", e)
        raise HTTPException(status_code=500, detail=f"Read failed: {str(e)}")


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
    logger.info("Starting AgenTest LLM Service v3.0.1")
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
        logger.info("  API Key: %s", "SET" if api_key else "NOT SET (optional)")

    elif provider == "openrouter":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        logger.info("  API Key: %s", "SET" if api_key else "NOT SET")
    
    elif provider == "llamacpp":
        if not LLAMACPP_AVAILABLE:
            logger.error("  llama-cpp-python NOT INSTALLED")
            logger.error("  Install with: pip install llama-cpp-python")
        else:
            logger.info("  llama-cpp-python available")
            logger.info("  Context window: %s", os.getenv("LLAMACPP_N_CTX", "4096"))
            logger.info("  GPU layers: %s", os.getenv("LLAMACPP_N_GPU_LAYERS", "0 (CPU only)"))

    elif provider == "openai":
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:8090/v1")
        logger.info("  OpenAI-compatible URL: %s", base_url)
        api_key = os.getenv("LLM_API_KEY", "")
        logger.info("  API Key: %s", "SET" if api_key else "NOT SET (optional)")


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