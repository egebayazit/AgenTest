from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
except Exception:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False


# ============================================================================
# AGENT MODE SYSTEM PROMPT - Simplified for direct instruction execution
# ============================================================================

AGENT_SYSTEM_PROMPT = """
You are a Windows UI automation agent. Execute the given instruction using elements from current_state.
OUTPUT: Valid JSON only. No markdown, no explanations.

COORDINATE SYSTEM:
Origin (0,0)=TOP-LEFT. X increases RIGHT. Y increases DOWN.

SCREEN REGIONS:
Compare element positions relative to others in the list:
- Small Y = top, Large Y = bottom
- Small X = left, Large X = right
"top-right" = element with largest X AND smallest Y
"bottom-left" = element with smallest X AND largest Y
"center" = element with mid-range X AND mid-range Y

SPATIAL REASONING:
"right of A" → X>Ax, pick min X, prefer similar Y
"left of A" → X<Ax, pick max X, prefer similar Y
"below A" → Y>Ay, pick min Y, prefer similar X
"above A" → Y<Ay, pick max Y, prefer similar X
"near A" → min distance

MATCHING PRIORITY:
1. Spatial/region words → calculate from anchor or check region bounds
2. No spatial → exact name match
3. No exact → partial match

RULES:
1. Use coords from STATE. Exception: explicit offset (e.g. "200px right") → calculate
2. "right of"/"left of" → same row (similar Y within 50px)
3. "below"/"above"/"" → same column (similar X within 50px)
4. Read note for additional hints

STATE: (x,y) [Type] Name | Value | AutomationID

ACTIONS:
click: {"type":"click","button":"left","click_count":1,"target":{"point":{"x":INT,"y":INT}}}
type: {"type":"type","text":"STR","enter":true|false}
key_combo: {"type":"key_combo","combo":["ctrl","c"]}
drag: {"type":"drag","from":{"x":INT,"y":INT},"to":{"x":INT,"y":INT}}
scroll: {"type":"scroll","delta":INT,"at":{"x":INT,"y":INT}} (+up,-down)
wait: {"type":"wait","ms":INT}

OUTPUT: {"reasoning":"...","action_id":"agent_N","coords_space":"physical","steps":[...]}

EXAMPLES:
Q: Click the Save button | State: Save(150,50), Cancel(200,50)
{"reasoning":"Save button at (150,50).","action_id":"agent_1","coords_space":"physical","steps":[{"type":"click","button":"left","click_count":1,"target":{"point":{"x":150,"y":50}}}]}

Q: Type hello and press Enter
{"reasoning":"Type+Enter.","action_id":"agent_2","coords_space":"physical","steps":[{"type":"type","text":"hello","enter":true}]}

Q: Ctrl+S
{"reasoning":"Shortcut.","action_id":"agent_3","coords_space":"physical","steps":[{"type":"key_combo","combo":["ctrl","s"]}]}
"""

# ============================================================================
# LOCAL AGENT SYSTEM PROMPT (for Ollama Qwen - Local Provider)
# ============================================================================

LOCAL_AGENT_SYSTEM_PROMPT = """You are a Windows UI automation agent. Execute the given instruction using elements from current_state.

OUTPUT: Valid JSON only. No markdown, no explanations.

CURRENT_STATE FORMAT:
ID | Name | Type | (x,y)
------------------------------------------------------------
1 | Save Settings | text | (10,20)
2 | Trash Bin | icon | (100,200)

CORE PRINCIPLE - SPATIAL IS KING:
If instruction says "next to", "near", "right of", "left of", "below", "above" -> PROXIMITY is PRIMARY.
- "right of A": Find elements where X > Ax AND Y similar to Ay (within 50px). Pick nearest.
- "left of A": Find elements where X < Ax AND Y similar to Ay (within 50px). Pick nearest.
- "below A": Find elements where Y > Ay AND X similar to Ax (within 50px). Pick nearest.
- "above A": Find elements where Y < Ay AND X similar to Ax (within 50px). Pick nearest.
If NO spatial words -> Use exact NAME match.

DUPLICATE NAMES:
If multiple elements have same name (e.g., two 'Input'):
- Use note spatial hints to pick the correct one
- "below X" = the one with Y > Xy
- "above X" = the one with Y < Xy

PRIORITY: Check note FIRST!
Read note for hints about which element to select. Apply spatial rules using those hints.

ACTION RULES:
1. 'click': Requires "target": {"point": {"x":..., "y":...}}.
   - Optional: "button": "right" for context menus.
   - Optional: "click_count": 2 for double-click.
2. 'type': Required: "text": "string". Optional: "enter": true.
3. 'key_combo': Format: "combo": ["ctrl", "c"] or ["enter"].
4. 'drag': Required: "from": {"x":..., "y":...}, "to": {"x":..., "y":...}.
5. 'scroll': Required: "delta": integer (+Up, -Down). Optional: "at": {"x":..., "y":...}.
6. 'wait': Required: "ms": integer.

CRITICAL: Use EXACT coordinates from current_state. Do not invent coordinates!
SINGLE TARGET: If instruction says "click X", find X and click ONLY X. Do not click other elements.

EXAMPLES:

EXAMPLE 1 — Click right of anchor:
Input: Click right of 'Settings' | State: Settings(100,50), Save(150,50), Cancel(200,100)
{
  "action_id": "agent_1",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":1,"target":{"point":{"x":150,"y":50}}}],
  "reasoning": "Anchor Settings at (100,50). Right=X>100, same Y. Save(150,50) nearest."
}

EXAMPLE 2 — Type with Enter:
{
  "action_id": "agent_2",
  "coords_space": "physical",
  "steps": [{"type":"type","text":"Hello","enter":true}],
  "reasoning": "Typed text and submitted."
}

EXAMPLE 3 — Key Combo:
{
  "action_id": "agent_3",
  "coords_space": "physical",
  "steps": [{"type":"key_combo","combo":["ctrl","s"]}],
  "reasoning": "Save shortcut."
}
"""


# Exceptions
class AgentBackendError(Exception): pass
class PlanParseError(AgentBackendError): pass
class SUTCommunicationError(AgentBackendError): pass
class LLMCommunicationError(AgentBackendError): pass


# Data classes
@dataclass
class AgentInstruction:
    """Agent mode instruction - simplified, no expected result"""
    instruction: str
    note: Optional[str] = None


@dataclass
class AgentActionLog:
    action_id: str
    plan: Dict[str, Any]
    ack: Dict[str, Any]
    state_before: Dict[str, Any]
    state_after: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=lambda: time.time())
    token_usage: Optional[Dict[str, Any]] = None


@dataclass
class AgentStepResult:
    status: str  # "executed", "failed", "stopped"
    actions: List[AgentActionLog]
    final_state: Optional[Dict[str, Any]]
    last_plan: Optional[Dict[str, Any]]
    reason: Optional[str] = None


@dataclass
class AgentScenarioOutcome:
    instruction: AgentInstruction
    result: AgentStepResult


@dataclass
class AgentScenarioResult:
    status: str  # "completed", "failed", "stopped"
    steps: List[AgentScenarioOutcome]
    final_state: Optional[Dict[str, Any]]
    reason: Optional[str] = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: str = ""


# ============================================================================
# AGENT MODE BACKEND
# ============================================================================

class AgentModeBackend:
    """
    Simplified backend for Agent Mode:
    - No semantic filtering
    - No pre-check
    - No expected result validation
    - Just execute instructions and report
    """
    
    def __init__(
        self,
        state_url_windriver: str,
        state_url_ods: str,
        action_url: str,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str = "http://localhost:11434",
        llm_api_key: Optional[str] = None,
        *,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        post_action_delay: float = 0.5,
        sut_timeout: float = 150.0,
        llm_timeout: float = 800.0,
        max_tokens: int = 1000,
        max_plan_steps: int = 10,
        http_referrer: str = "https://agentest.local/backend",
        client_title: str = "AgenTest Agent Mode Backend",
        enforce_json_response: bool = True,
        llamacpp_n_ctx: int = 4096,
        llamacpp_n_gpu_layers: int = 0,
        llamacpp_n_batch: int = 512
    ) -> None:
        
        self.state_url_windriver = state_url_windriver
        self.state_url_ods = state_url_ods
        self.llm_provider = llm_provider
        self.model = llm_model
        self.llm_base_url = llm_base_url.rstrip("/")
        self.api_key = llm_api_key
        self.enforce_json_response = enforce_json_response
        self.action_url = action_url
        self.system_prompt = system_prompt.strip()
        self.post_action_delay = post_action_delay
        self.sut_timeout = httpx.Timeout(sut_timeout)
        self.llm_timeout = httpx.Timeout(llm_timeout)
        self.max_tokens = max_tokens
        self.max_plan_steps = max_plan_steps
        self.http_referrer = http_referrer
        self.client_title = client_title
        self.llamacpp_n_ctx = llamacpp_n_ctx
        self.llamacpp_n_gpu_layers = llamacpp_n_gpu_layers
        self._llama_model: Optional[Llama] = None
        self.llamacpp_n_batch = llamacpp_n_batch
        
        if llm_provider == "llamacpp" and LLAMACPP_AVAILABLE:
            self._init_llamacpp_model()
    
    def _init_llamacpp_model(self) -> None:
        if not os.path.exists(self.model):
            raise FileNotFoundError(f"Model not found: {self.model}")
        
        try:
            self._llama_model = Llama(
                model_path=self.model,
                n_ctx=self.llamacpp_n_ctx,
                n_gpu_layers=self.llamacpp_n_gpu_layers,
                n_batch=self.llamacpp_n_batch,
                verbose=False
            )
            logger.info("Model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load llama.cpp: {e}")
    
    @classmethod
    def from_env(cls, **overrides) -> "AgentModeBackend":
        env = os.getenv
        return cls(
            state_url_windriver=overrides.get("state_url_windriver") or env("SUT_STATE_URL_WINDRIVER", "http://127.0.0.1:18800/state/for-llm"),
            state_url_ods=overrides.get("state_url_ods") or env("SUT_STATE_URL_ODS", "http://127.0.0.1:18800/state/from-ods"),
            action_url=overrides.get("action_url") or env("SUT_ACTION_URL", "http://192.168.137.52:18080/action"),
            llm_provider=overrides.get("llm_provider") or env("LLM_PROVIDER", "ollama"),
            llm_model=overrides.get("llm_model") or env("LLM_MODEL", "qwen2.5:7b-instruct-q6_k"),
            llm_base_url=overrides.get("llm_base_url") or env("LLM_BASE_URL", "http://localhost:11434"),
            llm_api_key=overrides.get("llm_api_key") or env("LLM_API_KEY"),
            **{k:v for k,v in overrides.items() if k not in ['state_url_windriver','state_url_ods','action_url','llm_provider','llm_model','llm_base_url','llm_api_key']}
        )
    
    # ========================================================================
    # SCENARIO & STEP EXECUTION 
    # ========================================================================
    
    async def run_scenario(
        self, 
        scenario_name: str, 
        instructions: List[AgentInstruction], 
        *, 
        temperature: float = 0.1,
        use_ods: bool = False,  # Default to WinDriver, can force ODS
        progress_callback: Optional[callable] = None,
        cancel_check: Optional[callable] = None
    ) -> AgentScenarioResult:
        
        if not instructions:
            raise ValueError("instructions required")
        
        # Helper to emit progress
        def emit(msg: str):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)
        
        outcomes = []
        final_state = None
        
        emit(f"STARTING AGENT SCENARIO: {scenario_name}")
        
        for idx, instr in enumerate(instructions, 1):
            # Check for cancellation before each step
            if cancel_check and cancel_check():
                emit("EXECUTION STOPPED BY USER")
                return AgentScenarioResult("stopped", outcomes, final_state, "User cancelled")
            
            emit(f"INSTRUCTION {idx}/{len(instructions)}: {instr.instruction}")
            
            result = await self.run_instruction(
                instr.instruction,
                instr.note,
                use_ods=use_ods,
                temperature=temperature,
                progress_callback=progress_callback,
                cancel_check=cancel_check
            )
            
            outcomes.append(AgentScenarioOutcome(instr, result))
            final_state = result.final_state
            
            # In agent mode, we don't fail the whole scenario on one instruction failure
            # We just continue to the next instruction
            if result.status == "stopped":
                return AgentScenarioResult("stopped", outcomes, final_state, "User cancelled")
        
        # Check if any instruction failed
        failed_count = sum(1 for o in outcomes if o.result.status == "failed")
        
        if failed_count == len(outcomes):
            return AgentScenarioResult("failed", outcomes, final_state, "All instructions failed")
        elif failed_count > 0:
            return AgentScenarioResult("completed", outcomes, final_state, f"{failed_count} instruction(s) failed")
        else:
            return AgentScenarioResult("completed", outcomes, final_state)
    
    async def run_instruction(
        self,
        instruction: str,
        note: Optional[str] = None,
        *,
        use_ods: bool = False,
        temperature: float = 0.1,
        progress_callback: Optional[callable] = None,
        cancel_check: Optional[callable] = None
    ) -> AgentStepResult:
        """
        Run a single agent instruction.
        No semantic filtering, no pre-check, no expected result validation.
        """
        
        # Helper to emit progress
        def emit(msg: str):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)
        
        actions_log = []
        state = {}
        
        # Check for cancellation
        if cancel_check and cancel_check():
            emit("EXECUTION STOPPED BY USER")
            return AgentStepResult("stopped", actions_log, state, None, "User cancelled")
        
        # Determine which state source to use
        method = "ODS" if use_ods else "WinDriver"
        emit(f"Fetching state via {method}...")
        
        try:
            state = await self._fetch_state(use_ods)
        except Exception as e:
            emit(f"State fetch failed: {e}")
            return AgentStepResult("failed", actions_log, state, None, f"State fetch error: {e}")
        
        # Plan
        emit("SENDING TO LLM...")
        
        # Check for cancellation before LLM request
        if cancel_check and cancel_check():
            emit("EXECUTION STOPPED BY USER")
            return AgentStepResult("stopped", actions_log, state, None, "User cancelled")
        
        try:
            # Create task for LLM request so we can poll for cancellation
            llm_task = asyncio.create_task(
                self._request_plan(
                    instruction=instruction,
                    note=note,
                    state=state,
                    temperature=temperature
                )
            )
            
            # Poll for cancellation while waiting for LLM
            while not llm_task.done():
                if cancel_check and cancel_check():
                    llm_task.cancel()
                    emit("EXECUTION STOPPED BY USER")
                    return AgentStepResult("stopped", actions_log, state, None, "User cancelled")
                await asyncio.sleep(0.5)
            
            plan, llm_view, token_usage = await llm_task
            
            # Emit token usage to UI
            if token_usage and token_usage.total_tokens > 0:
                emit(f"TOKEN USAGE [{token_usage.provider}]: Input={token_usage.input_tokens}, Output={token_usage.output_tokens}, Total={token_usage.total_tokens}")
        
        except asyncio.CancelledError:
            emit("EXECUTION STOPPED BY USER")
            return AgentStepResult("stopped", actions_log, state, None, "User cancelled")
        except Exception as e:
            emit(f"Planning failed: {e}")
            return AgentStepResult("failed", actions_log, state, None, f"LLM error: {e}")
        
        # Check for cancellation after LLM request
        if cancel_check and cancel_check():
            emit("EXECUTION STOPPED BY USER")
            return AgentStepResult("stopped", actions_log, state, None, "User cancelled")
        
        steps = plan.get("steps", [])
        
        if not steps:
            return AgentStepResult("failed", actions_log, state, plan, "No actions generated")
        
        emit(f"PARSED PLAN: {len(steps)} action(s)")
        
        # Execute
        emit("EXECUTING ACTION...")
        state_before = state
        
        # Check for cancellation before action
        if cancel_check and cancel_check():
            emit("EXECUTION STOPPED BY USER")
            return AgentStepResult("stopped", actions_log, state, None, "User cancelled")
        
        try:
            ack = await self._send_action(plan)
        except Exception as e:
            emit(f"Action failed: {e}")
            return AgentStepResult("failed", actions_log, state, plan, f"Action error: {e}")
        
        # Fetch final state
        final_state = await self._fetch_state_safe(state, use_ods)
        
        # Add llm_view to state_before copy for UI display
        state_before_with_llm = {**state_before, "llm_view": llm_view}
        
        # Convert token_usage to dict for serialization
        token_usage_dict = None
        if token_usage and token_usage.total_tokens > 0:
            token_usage_dict = {
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "total_tokens": token_usage.total_tokens,
                "provider": token_usage.provider
            }
        
        log = AgentActionLog(
            plan.get("action_id", ""),
            plan,
            ack,
            state_before_with_llm,
            final_state,
            token_usage=token_usage_dict
        )
        actions_log.append(log)
        
        # No validation in agent mode - just report execution
        emit("INSTRUCTION EXECUTED")
        return AgentStepResult("executed", actions_log, final_state, plan, "Executed successfully")
    
    async def _fetch_state(self, use_ods: bool = False) -> Dict:
        url = self.state_url_ods if use_ods else self.state_url_windriver
        async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
            resp = await client.post(url, json={})
            resp.raise_for_status()
            return resp.json()
    
    async def _fetch_state_safe(self, fallback: Dict, use_ods: bool = False) -> Dict:
        try:
            return await self._fetch_state(use_ods)
        except:
            return fallback
    
    async def _send_action(self, plan: Dict) -> Dict:
        async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
            resp = await client.post(self.action_url, json=plan)
            resp.raise_for_status()
            return resp.json()
    
    # ========================================================================
    # LLM Communication
    # ========================================================================
    
    async def _request_plan(
        self,
        *,
        instruction: str,
        note: Optional[str],
        state: Dict,
        temperature: float
    ) -> Tuple[Dict, str, TokenUsage]:
        
        messages, llm_view = self._build_messages(instruction, note, state)
        token_usage = TokenUsage()
        
        if self.llm_provider == "local":
            plan, token_usage = await self._request_plan_ollama(messages, temperature)
        elif self.llm_provider == "ollama":
            plan, token_usage = await self._request_plan_ollama(messages, temperature)
        elif self.llm_provider == "llamacpp":
            plan = await self._request_plan_llamacpp(messages, temperature)
        elif self.llm_provider == "openai":
            plan = await self._request_plan_openai(messages, temperature)
        elif self.llm_provider == "lmstudio":
            plan = await self._request_plan_openai(messages, temperature)
        elif self.llm_provider == "anthropic":
            plan, token_usage = await self._request_plan_anthropic(messages, temperature)
        elif self.llm_provider == "gemini":
            plan = await self._request_plan_gemini(messages, temperature)
        elif self.llm_provider == "custom":
            plan = await self._request_plan_openai(messages, temperature)
        else:
            plan = await self._request_plan_openrouter(messages, temperature)
        
        return plan, llm_view, token_usage
    
    async def _request_plan_ollama(self, messages: List, temperature: float) -> Tuple[Dict, TokenUsage]:
        """Ollama API with JSON mode."""
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            resp = await client.post(
                f"{self.llm_base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature}
                }
            )
            data = resp.json()
            
            # Extract token usage from Ollama response
            token_usage = TokenUsage(provider="Local/Ollama")
            if "prompt_eval_count" in data:
                token_usage.input_tokens = data.get("prompt_eval_count", 0)
            if "eval_count" in data:
                token_usage.output_tokens = data.get("eval_count", 0)
            token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens
            
            return self._parse_plan(data["message"]["content"]), token_usage

    async def _request_plan_llamacpp(self, messages: List, temperature: float) -> Dict:
        if not self._llama_model:
            raise LLMCommunicationError("Model not loaded")
        
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self._llama_model.create_completion(
                prompt,
                max_tokens=self.max_tokens,
                temperature=temperature
            )
        )
        return self._parse_plan(resp["choices"][0]["text"])
    
    async def _request_plan_openai(self, messages: List, temperature: float) -> Dict:
        """OpenAI-compatible API with JSON mode."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            resp = await client.post(
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            return self._parse_plan(resp.json()["choices"][0]["message"]["content"])
    
    async def _request_plan_openrouter(self, messages: List, temperature: float) -> Dict:
        """OpenRouter API with JSON mode."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.http_referrer
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            return self._parse_plan(resp.json()["choices"][0]["message"]["content"])
    
    async def _request_plan_anthropic(self, messages: List, temperature: float) -> Tuple[Dict, TokenUsage]:
        """Anthropic Claude API support."""
        if not ANTHROPIC_AVAILABLE:
            raise LLMCommunicationError("anthropic package not installed")
        
        # Extract system prompt and user message
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m.get("role") != "system"
        ]
        
        client = AsyncAnthropic(api_key=self.api_key)
        
        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=sys_msg,
            messages=user_messages,
            temperature=temperature
        )
        
        # Extract token usage
        token_usage = TokenUsage(provider="Anthropic")
        if hasattr(response, 'usage') and response.usage:
            token_usage.input_tokens = response.usage.input_tokens
            token_usage.output_tokens = response.usage.output_tokens
            token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens
        
        # Extract text content from response
        content = response.content[0].text if response.content else ""
        return self._parse_plan(content), token_usage
    
    async def _request_plan_gemini(self, messages: List, temperature: float) -> Dict:
        """Google Gemini API support with structured JSON output."""
        if not GEMINI_AVAILABLE:
            raise LLMCommunicationError("google-genai package not installed")
        
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        
        client = genai.Client(api_key=self.api_key)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.model,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=sys_msg,
                    temperature=temperature,
                    max_output_tokens=max(self.max_tokens, 8192),
                    response_mime_type="application/json",
                ),
            ),
        )
        
        # Check candidates
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
            logger.info(f"  Finish Reason: {finish_reason}")
        else:
            logger.warning("  No candidates in response!")
            logger.info(f"  Full response: {response}")
        
        # Extract text content from response
        content = response.text if response.text else ""
        
        if not content or len(content.strip()) < 10:
            raise LLMCommunicationError("Gemini returned empty/incomplete response.")
        
        return self._parse_plan(content)
    
    def _build_messages(
        self,
        instruction: str,
        note: Optional[str],
        state: Dict
    ) -> Tuple[List[Dict], str]:
        
        elements_to_show = state.get("elements", [])
        
        # No semantic filtering in agent mode - send ALL elements
        
        # Build LLM View
        lines = []
        lines.append("Format: (x,y) [Type] Name | Value | AutomationID")
        lines.append("-" * 60)
        
        for el in elements_to_show:
            name = el.get("name", "NoName").replace("|", "")
            e_type = el.get("type", "unknown")
            center = el.get("center", {})
            x, y = center.get("x", 0), center.get("y", 0)
            value = el.get("value", "")
            automation_id = el.get("automation_id", "")
    
            value_str = f"Value: {value}" if value else ""
            id_str = f"ID: {automation_id}" if automation_id else ""
    
            lines.append(f"({x},{y}) [{e_type}] {name} | {value_str} | {id_str}")
            
        llm_view = "\n".join(lines)
        
        # DEBUG LOG: LLM'e giden nihai görüntü
        logger.info("=" * 80)
        logger.info("AGENT MODE - SENDING TO LLM")
        logger.info("=" * 80)
        logger.info("Instruction: %s", instruction)
        logger.info("Note: %s", note)
        logger.info("=" * 80)
        logger.info("Current View:\n%s", llm_view)
        logger.info("=" * 80)
        
        # Simplified payload for agent mode
        payload = {
            "instruction": instruction,
            "current_state": llm_view
        }
        
        if note:
            payload["note"] = note
        
        # Select prompt based on provider type
        if self.llm_provider == "local":
            active_prompt = LOCAL_AGENT_SYSTEM_PROMPT.strip()
        else:
            active_prompt = self.system_prompt
        
        messages = [
            {"role": "system", "content": active_prompt},
            {"role": "user", "content": json.dumps(payload)}
        ]
        
        return messages, llm_view
    
    def _parse_plan(self, content: str) -> Dict:
        # DEBUG LOG: Raw LLM response
        logger.info("=" * 80)
        logger.info("AGENT MODE - RECEIVED FROM LLM")
        logger.info("=" * 80)
        if len(content) > 1500:
            logger.info("%s\n... (truncated, %d total chars)", content[:1500], len(content))
        else:
            logger.info("%s", content)
        logger.info("=" * 80)
        
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        
        try:
            plan = json.loads(text)
        except Exception as e:
            logger.error("JSON PARSE ERROR: %s", str(e))
            logger.error("Cleaned text was: %s", text[:500])
            raise PlanParseError("Invalid JSON")
        
        if "steps" not in plan:
            plan["steps"] = []
        if "action_id" not in plan:
            plan["action_id"] = f"agent_{int(time.time())}"
        if "coords_space" not in plan:
            plan["coords_space"] = "physical"
        
        # DEBUG LOG: Parsed plan
        logger.info("=" * 80)
        logger.info("AGENT MODE - PARSED PLAN")
        logger.info("=" * 80)
        logger.info("Action ID: %s", plan.get("action_id"))
        logger.info("Reasoning: %s", plan.get("reasoning", "N/A"))
        logger.info("Steps: %d", len(plan.get("steps", [])))
        for idx, step in enumerate(plan.get("steps", []), 1):
            step_type = step.get("type", "unknown")
            if step_type == "click":
                pt = step.get("target", {}).get("point", {})
                logger.info("  %d. Click at (%d, %d)", idx, pt.get("x", -1), pt.get("y", -1))
            elif step_type == "type":
                logger.info("  %d. Type: '%s'", idx, step.get("text", ""))
            elif step_type == "key_combo":
                logger.info("  %d. Key combo: %s", idx, "+".join(step.get("combo", [])))
            elif step_type == "drag":
                logger.info("  %d. Drag from (%d,%d) to (%d,%d)", idx,
                          step.get("from", {}).get("x", -1), step.get("from", {}).get("y", -1),
                          step.get("to", {}).get("x", -1), step.get("to", {}).get("y", -1))
            elif step_type == "scroll":
                logger.info("  %d. Scroll delta=%d", idx, step.get("delta", 0))
            elif step_type == "wait":
                logger.info("  %d. Wait %dms", idx, step.get("ms", 0))
            else:
                logger.info("  %d. %s", idx, step_type)
        logger.info("=" * 80)
        
        return plan
