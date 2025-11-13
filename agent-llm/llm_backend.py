# llm_backend.py
# Core LLM orchestration for AgenTest (Controller handles filtering & plan safety)

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

DEFAULT_SYSTEM_PROMPT = (
    "You are the AgenTest action planner. Your job is to convert a manual UI testing step into a precise plan "
    "for the Windows System Under Test (SUT). Always obey the schema and rules below.\n\n"
    "INPUTS: (a) test step, (b) expected result, (c) CURRENT UI STATE (elements with rects), "
    "(d) RECENT ACTIONS already executed (do NOT repeat them unless the state shows they failed), (e) SCREEN size.\n\n"
    "HARD RULES:\n"
    "- Output MUST be ONE JSON OBJECT. No markdown fences.\n"
    "- Output MUST be entirely in JSON format; do NOT include any prose/explanations/comments outside the JSON.\n"
    "- Use ONLY elements from state; do NOT invent coordinates or rectangles.\n"
    "- Click at the CENTER of the provided rect.\n"
    "- Prefer enabled + active-window elements. If nothing matches, use keyboard-only plan.\n"
    "- Do NOT relaunch or reopen something that is already open/active.\n"
    "- If the expected result already holds, return an empty 'steps' array.\n"
    "- Limit the 'steps' array to at most 24 actions; if more are needed, explain why in 'reasoning' and stop.\n"
    "- Avoid repeating identical clicks/moves; only include retries if the state shows they are required.\n"
    "- Output MUST include a 'steps' array matching the schema; only return [] when the expected result already holds.\n"
    "- ALL coordinates/rects MUST be within the provided screen bounds; rect width/height MUST be > 2.\n\n"
    "OUTPUT SCHEMA:\n"
    "{\n"
    '  "action_id": "step_<timestamp>",\n'
    '  "coords_space": "physical",\n'
    '  "steps": [],\n'
    '  "reasoning": "short explanation"\n'
    "}\n\n"
    +    "ALLOWED ACTIONS:\n"
   '- CLICK:  {"type":"click","button":"left|right|middle","click_count":1,\n'
    '           "modifiers":["ctrl","shift","alt","win"],\n'
    '           "target":{"rect":{"x":100,"y":100,"w":80,"h":30}}}\n'
   "  or\n"
    '- CLICK:  {"type":"click","button":"left|right|middle","click_count":1,\n'
    '           "target":{"point":{"x":960,"y":540}}}\n'
    '- TYPE:   {"type":"type","text":"...","delay_ms":30,"enter":false}\n'
    '- KEY_COMBO: {"type":"key_combo","combo":["win","s"]}\n'
    '- DRAG:   {"type":"drag","from":{"x":..,"y":..},"to":{"x":..,"y":..},"button":"left","hold_ms":120}\n'
    '- SCROLL: {"type":"scroll","delta":-240,"horizontal":false,"at":{"x":800,"y":600}}\n'
    '- MOVE:   {"type":"move","point":{"x":..,"y":..},"settle_ms":150}\n'
    '- WAIT:   {"type":"wait","ms":300}\n\n'
    "TIMING HINTS:\n"
    "- After opening or changing views, WAIT ~800-1500ms before the next step.\n"
    "- Before TYPE after CLICK/KEY_COMBO that focuses an input, WAIT ~200-400ms.\n"
    "- Use per-character delays for TYPE (delay_ms about 30-40).\n\n"
    "SELECTION RULES:\n"
    "1) Search all elements above and pick the best match.\n"
    "2) If no element clearly matches, produce a keyboard-only plan to progress.\n\n"
    "DO NOTS:\n"
    "- Do not output anything except the JSON object.\n"
    "- Do not send Enter as a separate key_combo when TYPE.enter=true is available.\n\n"
   "CLICK TARGET RULE:\n"
    "- For CLICK you may use either target.rect (click center of rect) or target.point (exact coordinate). "
    "Do NOT invent coordinates; derive them only from the provided element rects/centers in current_state.\n\n"
     "# TOKEN SAFETY\n"
    "# TOKEN SAFETY\n"
    "- Be concise; avoid redundant fields or deep nesting.\n"
    "- If you approach token limits, summarise or drop low-value details.\n\n"
    "# OUTPUT VALIDATION\n"
    "- If the expected result already holds, output an empty steps array with short reasoning.\n"
    "- Otherwise produce the shortest plan that advances toward it.\n\n"
    "# RECOVERY BEHAVIOR\n"
    "- If your previous reply was cut off or invalid, resend only the compact JSON plan (no explanations).\n"
)

AGEN_TEST_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action_id", "coords_space", "steps", "reasoning"],
    "properties": {
        "action_id": {"type": "string"},
        "coords_space": {"type": "string", "enum": ["physical"]},
        "reasoning": {"type": "string", "maxLength": 800},
        "steps": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "oneOf": [
                    {  # CLICK
                        "properties": {
                            "type": {"const": "click"},
                            "button": {"enum": ["left", "right", "middle"]},
                            "click_count": {"type": "integer", "minimum": 1, "maximum": 2},
                            "modifiers": {
                                "type": "array",
                                "items": {"enum": ["ctrl", "shift", "alt", "win"]},
                                "uniqueItems": True,
                            },
                            "target": {
                                "type": "object",
                                "properties": {
                                    "rect": {
                                        "type": "object",
                                        "required": ["x", "y", "w", "h"],
                                        "properties": {
                                            "x": {"type": "number"},
                                            "y": {"type": "number"},
                                            "w": {"type": "number"},
                                            "h": {"type": "number"},
                                        },
                                        "additionalProperties": False,
                                    }
                                },
                                "required": ["rect"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["type", "button", "click_count", "target"],
                    },
                    {  # CLICK  (target.rect OR target.point)
                        "properties": {
                            "type": {"const": "click"},
                            "button": {"enum": ["left", "right", "middle"]},
                            "click_count": {"type": "integer", "minimum": 1, "maximum": 2},
                            "modifiers": {
                                "type": "array",
                                "items": {"enum": ["ctrl", "shift", "alt", "win"]},
                                "uniqueItems": True,
                            },
                            "target": {
                                "type": "object",
                                "additionalProperties": False,
                                "oneOf": [
                                    {   # rect hedefi
                                        "properties": {
                                            "rect": {
                                                "type": "object",
                                                "required": ["x", "y", "w", "h"],
                                                "properties": {
                                                    "x": {"type": "number"},
                                                    "y": {"type": "number"},
                                                    "w": {"type": "number"},
                                                    "h": {"type": "number"},
                                                },
                                                "additionalProperties": False,
                                            }
                                        },
                                        "required": ["rect"]
                                    },
                                    {   # point hedefi
                                        "properties": {
                                            "point": {
                                                "type": "object",
                                                "required": ["x", "y"],
                                                "properties": {
                                                    "x": {"type": "number"},
                                                    "y": {"type": "number"},
                                                },
                                                "additionalProperties": False,
                                            }
                                        },
                                        "required": ["point"]
                                    }
                                ],
                            },
                        },
                        "required": ["type", "button", "click_count", "target"],
                    },
                ],
            },
        },
    },
}

class BackendError(Exception): ...
class PlanParseError(BackendError): ...
class SUTCommunicationError(BackendError): ...
class LLMCommunicationError(BackendError): ...

@dataclass
class StepDefinition:
    test_step: str
    expected_result: str
    note_to_llm: Optional[str] = None

@dataclass
class ActionExecutionLog:
    action_id: str
    plan: Dict[str, Any]
    ack: Dict[str, Any]
    state_before: Dict[str, Any]
    state_after: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=lambda: time.time())

@dataclass
class RunResult:
    status: str
    attempts: int
    actions: List[ActionExecutionLog]
    final_state: Optional[Dict[str, Any]]
    last_plan: Optional[Dict[str, Any]]
    reason: Optional[str] = None

@dataclass
class ScenarioStepOutcome:
    step: StepDefinition
    result: RunResult

@dataclass
class ScenarioResult:
    status: str
    steps: List[ScenarioStepOutcome]
    final_state: Optional[Dict[str, Any]]
    reason: Optional[str] = None

class LLMBackend:
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    _SCREENSHOT_KEYS = {"screenshot", "screenshot_png", "screenshot_jpg", "screenshot_base64", "b64"}

    def __init__(
        self,
        state_url: str,
        action_url: str,
        openrouter_api_key: str,
        openrouter_model: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_attempts: int = 6,
        post_action_delay: float = 1.0,
        sut_timeout: float = 50.0,
        llm_timeout: float = 50.0,
        max_tokens: int = 800,
        max_plan_steps: int = 24,
        schema_retry_limit: int = 2,
        http_referrer: str = "https://agetest.local/backend",
        client_title: str = "AgenTest LLM Backend",
        omit_large_blobs: bool = True,
        enforce_json_response: bool = True,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not openrouter_api_key:
            raise ValueError("openrouter_api_key is required")
        if not openrouter_model:
            raise ValueError("openrouter_model is required")
        if max_plan_steps <= 0:
            raise ValueError("max_plan_steps must be positive")
        if schema_retry_limit < 0:
            raise ValueError("schema_retry_limit must be >= 0")

        self.state_url = state_url
        self.enforce_json_response = enforce_json_response
        self._json_response_enabled = enforce_json_response
        self.action_url = action_url
        self.api_key = openrouter_api_key
        self.model = openrouter_model
        self.system_prompt = system_prompt.strip()
        self.max_attempts = max_attempts
        self.post_action_delay = post_action_delay
        self.sut_timeout = httpx.Timeout(sut_timeout)
        self.llm_timeout = httpx.Timeout(llm_timeout)
        self.max_tokens = max_tokens
        self.max_plan_steps = max_plan_steps
        self.schema_retry_limit = schema_retry_limit
        self.http_referrer = http_referrer
        self.client_title = client_title
        self.omit_large_blobs = omit_large_blobs
        self.json_schema = json_schema

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMBackend":
        env = os.getenv

        def _env_bool(name: str, default: str = "1") -> bool:
            raw = overrides.pop(name, None)
            if raw is None:
                raw = overrides.pop(name.lower(), None)
            if raw is not None:
                if isinstance(raw, str):
                    return raw.lower() not in {"0", "false", "no", ""}
                return bool(raw)
            value = env(name, default)
            if value is None:
                return False
            if isinstance(value, str):
                return value.lower() not in {"0", "false", "no", ""}
            return bool(value)

        params = {
            "state_url": overrides.pop("state_url", env("SUT_STATE_URL", "http://127.0.0.1:18800/state/for-llm")),
            "action_url": overrides.pop("action_url", env("SUT_ACTION_URL", "http://192.168.137.249:18080/action")),
            "openrouter_api_key": overrides.pop("openrouter_api_key", env("OPENROUTER_API_KEY", "")),
            "openrouter_model": overrides.pop("openrouter_model", env("OPENROUTER_MODEL", "")),
            "enforce_json_response": _env_bool("ENFORCE_JSON_RESPONSE"),
        }
        params.update(overrides)
        return cls(**params)

    async def run_scenario(
        self,
        steps: List[StepDefinition],
        *,
        temperature: float = 0.1,
        max_attempts: Optional[int] = None,
    ) -> ScenarioResult:
        if not steps:
            raise ValueError("steps must contain at least one StepDefinition")

        history: List[Dict[str, Any]] = []
        outcomes: List[ScenarioStepOutcome] = []
        final_state: Optional[Dict[str, Any]] = None
        max_per_step = max_attempts or self.max_attempts

        for step in steps:
            result = await self.run_step(
                test_step=step.test_step,
                expected_result=step.expected_result,
                note_to_llm=step.note_to_llm,
                max_attempts=max_per_step,
                recent_actions=history,
                temperature=temperature,
            )
            outcomes.append(ScenarioStepOutcome(step=step, result=result))
            final_state = result.final_state
            for log in result.actions:
                history.append(self._summarise_for_prompt(log))
            if result.status != "passed":
                return ScenarioResult(status=result.status, steps=outcomes, final_state=final_state, reason=result.reason)

        return ScenarioResult(status="passed", steps=outcomes, final_state=final_state)

    async def run_step(
        self,
        test_step: str,
        expected_result: str,
        note_to_llm: Optional[str] = None,
        *,
        max_attempts: Optional[int] = None,
        recent_actions: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
    ) -> RunResult:
        attempt_limit = max_attempts or self.max_attempts
        history_payload: List[Dict[str, Any]] = list(recent_actions or [])
        actions_log: List[ActionExecutionLog] = []
        last_plan: Optional[Dict[str, Any]] = None

        state = await self._fetch_state()
        logger.info("State for LLM (/state/for-llm):\n%s", json.dumps(state, indent=2, ensure_ascii=False))

        final_state: Optional[Dict[str, Any]] = state

        for attempt in range(1, attempt_limit + 1):
            schema_hint: Optional[str] = None
            for schema_attempt in range(self.schema_retry_limit + 1):
                plan = await self._request_plan(
                    test_step=test_step,
                    expected_result=expected_result,
                    note_to_llm=note_to_llm,
                    state=state,
                    recent_actions=history_payload,
                    temperature=temperature,
                    schema_hint=schema_hint,
                )

                # Validate plan geometry against screen; if invalid, ask LLM to fix with a concrete hint
                validation_error = self._validate_plan_against_screen(plan, state)
                if validation_error:
                    if schema_attempt >= self.schema_retry_limit:
                        msg = f"Plan failed validation: {validation_error}"
                        logger.error(msg)
                        return RunResult(
                            status="error",
                            attempts=attempt,
                            actions=actions_log,
                            final_state=state,
                            last_plan=plan,
                            reason=msg,
                        )
                    schema_hint = (
                        f"Your previous plan had this error: {validation_error}. "
                        "Return a valid JSON object with a 'steps' array that obeys the schema and screen bounds. "
                        "Do NOT invent rectangles. Use ONLY element rects provided in current_state."
                    )
                    continue

                if not plan.pop("_backend_steps_substituted", False):
                    pass
                else:
                    if schema_attempt >= self.schema_retry_limit:
                        message = "LLM plan missing valid 'steps'; request aborted."
                        logger.error(message)
                        return RunResult(
                            status="error",
                            attempts=attempt,
                            actions=actions_log,
                            final_state=state,
                            last_plan=plan,
                            reason=message,
                        )
                    schema_hint = (
                        "Previous response violated the JSON schema. "
                        "Return a valid object with a 'steps' array (use [] only when the expected result already holds)."
                    )
                    continue

                # good plan
                break
            else:
                raise RuntimeError("Schema retry loop exited unexpectedly")

            last_plan = plan
            steps = plan.get("steps", [])

            # --- CONDITIONAL EMPTY-STEPS HANDLING ---
            if not steps:
                if self._expected_holds(state, expected_result):
                    reason = plan.get("reasoning") or "Expected result observed in current state."
                    return RunResult(
                        status="passed",
                        attempts=attempt,
                        actions=actions_log,
                        final_state=state,
                        last_plan=plan,
                        reason=reason,
                    )
                else:
                    # Expected not observed, keep iterating to get a real plan
                    last_plan = plan
                    continue

            ack = await self._send_action(plan)
            log_entry = ActionExecutionLog(
                action_id=plan.get("action_id", ""),
                plan=plan,
                ack=ack,
                state_before=state,
            )
            actions_log.append(log_entry)
            history_payload.append(self._summarise_for_prompt(log_entry))

            if ack.get("status") != "ok":
                final_state = await self._fetch_state_safe(state)
                log_entry.state_after = final_state
                message = f"SUT /action returned non-ok status: {ack}"
                logger.warning(message)
                return RunResult(
                    status="error",
                    attempts=attempt,
                    actions=actions_log,
                    final_state=final_state,
                    last_plan=plan,
                    reason=message,
                )

            await asyncio.sleep(self.post_action_delay)
            state = await self._fetch_state()
            logger.info("State for LLM (after action):\n%s", json.dumps(state, indent=2, ensure_ascii=False))
            log_entry.state_after = state
            final_state = state

        return RunResult(
            status="failed",
            attempts=attempt_limit,
            actions=actions_log,
            final_state=final_state,
            last_plan=last_plan,
            reason="Planner exhausted max attempts without reaching expected result.",
        )

    async def _fetch_state(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
                response = await client.post(self.state_url, json={})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            message = f"Failed to contact SUT /state at {self.state_url}: {exc}"
            logger.error(message)
            raise SUTCommunicationError(message) from exc

        try:
            state = response.json()
        except json.JSONDecodeError as exc:
            message = "SUT /state returned invalid JSON."
            logger.error(message)
            raise SUTCommunicationError(message) from exc

        if self.omit_large_blobs and isinstance(state, dict):
            state = self._prune_state_blobs(state)
        return state

    async def _fetch_state_safe(self, fallback: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await self._fetch_state()
        except BackendError:
            return fallback

    async def _send_action(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
                response = await client.post(self.action_url, json=plan)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            message = f"Failed to contact SUT /action at {self.action_url}: {exc}"
            logger.error(message)
            raise SUTCommunicationError(message) from exc

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            message = "SUT /action returned invalid JSON."
            logger.error(message)
            raise SUTCommunicationError(message) from exc

    async def _request_plan(
        self,
        *,
        test_step: str,
        expected_result: str,
        note_to_llm: Optional[str],
        state: Dict[str, Any],
        recent_actions: List[Dict[str, Any]],
        temperature: float,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = self._build_messages(
            test_step=test_step,
            expected_result=expected_result,
            note_to_llm=note_to_llm,
            state=state,
            recent_actions=recent_actions,
            schema_hint=schema_hint,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.http_referrer,
            "X-Title": self.client_title,
            "Content-Type": "application/json",
        }

        use_json_modes = [self._json_response_enabled]
        if self._json_response_enabled:
            use_json_modes.append(False)

        last_error: Optional[str] = None

        for enforce_json in use_json_modes:
            body = self._build_request_body(messages, temperature, enforce_json)
            try:
                async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                    response = await client.post(self.OPENROUTER_URL, headers=headers, json=body)
            except httpx.HTTPError as exc:
                message = f"OpenRouter request failed: {exc}"
                logger.error(message)
                raise LLMCommunicationError(message) from exc

            if response.status_code >= 400:
                response_text = response.text
                last_error = f"OpenRouter returned {response.status_code}: {response_text}"
                if enforce_json and ("JSON Schema Validation Error" in response_text or "response_format is not supported" in response_text):
                    logger.warning("Model/provider rejected response_format; retrying without enforced JSON.")
                    self._json_response_enabled = False
                    continue
                logger.error(last_error)
                raise LLMCommunicationError(last_error)

            data = response.json()
            logger.warning("Raw LLM response: %r", data)
            try:
                content = self._extract_llm_content(data)
            except PlanParseError:
                # One-shot simple retry with smaller max_tokens and slightly lower temperature
                if self.max_tokens > 400:
                    saved = self.max_tokens
                    self.max_tokens = 400
                    try:
                        body2 = self._build_request_body(messages, max(0.0, temperature - 0.05), self._json_response_enabled)
                        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                            resp2 = await client.post(self.OPENROUTER_URL, headers=headers, json=body2)
                            resp2.raise_for_status()
                        data2 = resp2.json()
                        logger.warning("Raw LLM response (retry): %r", data2)
                        content = self._extract_llm_content(data2)
                    finally:
                        self.max_tokens = saved
                else:
                    raise

            plan = self._parse_plan(content)
            self._json_response_enabled = bool(enforce_json)
            return plan

        if last_error:
            raise LLMCommunicationError(last_error)
        raise LLMCommunicationError("OpenRouter request failed without response body")

    def _build_request_body(self, messages: List[Dict[str, Any]], temperature: float, enforce_json: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "provider": {
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            # Cut off common junk tails early
            "stop": ["</JSON_ONLY>", "<|end|>", "assistant:", "commentary to=assistant"],
        }
        if enforce_json and self.enforce_json_response:
            if self.json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "AgenTestPlan", "schema": self.json_schema},
                }
            else:
                body["response_format"] = {"type": "json_object"}
        return body

    def _build_messages(
        self,
        *,
        test_step: str,
        expected_result: str,
        note_to_llm: Optional[str],
        state: Dict[str, Any],
        recent_actions: List[Dict[str, Any]],
        schema_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Use controller-provided state directly (controller is responsible for normalization)
        clean_state = dict(state) if isinstance(state, dict) else state

        screen_info = state.get("screen", {})
        payload: Dict[str, Any] = {
            "test_step": test_step,
            "expected_result": expected_result,
            "screen": {
                "w": screen_info.get("w"),
                "h": screen_info.get("h"),
                "dpiX": screen_info.get("dpiX"),
                "dpiY": screen_info.get("dpiY"),
            },
            "current_state": clean_state,
            "recent_actions": recent_actions,
        }
        if schema_hint:
            payload["backend_schema_hint"] = schema_hint
        if note_to_llm:
            payload["note_to_llm"] = note_to_llm

        user_content = json.dumps(payload, ensure_ascii=False)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"<JSON_ONLY>{user_content}</JSON_ONLY>"},
        ]

    def _extract_llm_content(self, data: Dict[str, Any]) -> str:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, AttributeError) as exc:
            message = f"Unexpected OpenRouter payload: {data}"
            logger.error(message)
            raise LLMCommunicationError(message) from exc

        raw_content = msg.get("content")
        if isinstance(raw_content, str):
            content = raw_content.strip()
        elif raw_content is None:
            content = ""
        else:
            content = str(raw_content).strip()

        fallback_source = None
        if not content:
            reasoning = msg.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning.strip()
                fallback_source = "reasoning"

        if not content:
            details = msg.get("reasoning_details")
            if isinstance(details, list):
                collected = []
                for item in details:
                    if isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str) and txt.strip():
                            collected.append(txt.strip())
                if collected:
                    content = " ".join(collected)
                    fallback_source = fallback_source or "reasoning_details"

        if not content:
            raise PlanParseError("LLM returned empty content.")

        if fallback_source:
            logger.warning("LLM content missing; using %s field as fallback.", fallback_source)
        return content

    # -------------------- Robust JSON plan parsing --------------------
    def _parse_plan(self, content: str) -> Dict[str, Any]:
        stripped = self._strip_common_wrappers(content).strip()
        if not stripped:
            raise PlanParseError("LLM returned blank plan.")

        try:
            plan = json.loads(stripped)
        except json.JSONDecodeError:
            candidate = self._extract_first_json_object_with_keys(
                stripped,
                required_keys=("coords_space", "steps"),
                alt_required=("action_id",),
            )
            if candidate is None:
                repaired = self._attempt_repair_plan(stripped)
                if repaired is None:
                    logger.error("LLM plan is not valid JSON: %s...", stripped[:200])
                    raise PlanParseError(f"LLM plan is not valid JSON: {stripped[:200]}...")
                plan = json.loads(repaired)
            else:
                plan = candidate

        if not isinstance(plan, dict):
            raise PlanParseError("LLM plan must be a JSON object.")

        steps, steps_source = self._ensure_step_list(plan)
        if steps is None:
            steps_raw = plan.get("steps")
            logger.warning("LLM plan missing valid 'steps'; substituting empty list. steps_type=%s", type(steps_raw).__name__)
            logger.debug("Raw LLM content (trimmed): %s", stripped[:512])
            plan["steps"] = []
            reasoning = plan.get("reasoning")
            note = "Backend replaced invalid 'steps' with empty list due to parsing failure."
            plan["_backend_steps_substituted"] = True
            plan["reasoning"] = f"{reasoning} | {note}" if reasoning else note
            steps = plan["steps"]
        else:
            plan["steps"] = steps
            if steps_source and steps_source != "steps":
                logger.warning("LLM plan missing top-level 'steps'; recovered from %s", steps_source)
                reasoning = plan.get("reasoning")
                note = f"Backend recovered steps from {steps_source}."
                plan["reasoning"] = f"{reasoning} | {note}" if reasoning else note

        if self.max_plan_steps and len(steps) > self.max_plan_steps:
            logger.warning("LLM plan returned %s steps; truncating to backend cap of %s", len(steps), self.max_plan_steps)
            plan["steps"] = steps[: self.max_plan_steps]
            steps = plan["steps"]
            reasoning = plan.get("reasoning")
            note = f"Truncated to first {self.max_plan_steps} steps by backend."
            plan["reasoning"] = f"{reasoning} | {note}" if reasoning else note

        if "action_id" not in plan or not plan["action_id"]:
            plan["action_id"] = f"step_{int(time.time() * 1000)}"
        if "coords_space" not in plan:
            plan["coords_space"] = "physical"

        return plan

    def _strip_common_wrappers(self, text: str) -> str:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
        text = re.sub(r'^\s*(?:assistant:|assistant\s*->\s*|commentary\s*:)\s*', "", text, flags=re.IGNORECASE)
        return text

    def _extract_first_json_object_with_keys(
        self,
        text: str,
        required_keys: Tuple[str, ...],
        alt_required: Tuple[str, ...] = (),
        max_scan: int = 3,
    ) -> Optional[Dict[str, Any]]:
        junk_idx = text.find("commentary to=assistant")
        if junk_idx != -1:
            brace_after = text.find("{", junk_idx)
            if brace_after != -1:
                text = text[brace_after:]
        starts = [m.start() for m in re.finditer(r"{", text)]
        scanned = 0
        for s in starts:
            if scanned >= max_scan:
                break
            scanned += 1
            candidate = self._slice_balanced_json(text, s)
            if candidate is None:
                continue
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                has_required = all(k in obj for k in required_keys)
                has_alt = all(k in obj for k in alt_required)
                if has_required or has_alt:
                    return obj
        return None

    def _slice_balanced_json(self, text: str, start_idx: int) -> Optional[str]:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text[start_idx:], start=start_idx):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
        return None

    def _ensure_step_list(self, plan: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        steps = self._normalise_step_candidate(plan.get("steps"))
        if steps is not None:
            return steps, "steps"
        return self._recover_steps_from_plan(plan)

    def _recover_steps_from_plan(self, plan: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        candidates: List[Tuple[str, Any]] = []
        for key in ("actions", "plan_steps", "planSteps", "steps_plan", "operations"):
            candidates.append((key, plan.get(key)))
        for container_key in ("plan", "action_plan", "result", "data", "payload"):
            container = plan.get(container_key)
            if isinstance(container, dict):
                for nested_key in ("steps", "actions"):
                    candidates.append((f"{container_key}.{nested_key}", container.get(nested_key)))
        for source, candidate in candidates:
            normalised = self._normalise_step_candidate(candidate)
            if normalised is not None:
                return normalised, source
        return None, None

    def _normalise_step_candidate(self, value: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
        return None

    def _summarise_for_prompt(self, entry: ActionExecutionLog) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "action_id": entry.action_id,
            "steps": entry.plan.get("steps", []),
            "ack": entry.ack,
            "timestamp": entry.timestamp,
        }
        if entry.state_after is not None:
            summary["result_state_hint"] = {
                "status": entry.state_after.get("status"),
                "focused_element": entry.state_after.get("focused_element"),
            }
        return summary

    # ---------- Plan validation vs screen ----------
    def _validate_plan_against_screen(self, plan: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
        screen = state.get("screen") or {}
        sw, sh = screen.get("w"), screen.get("h")
        if not isinstance(sw, int) or not isinstance(sh, int) or sw <= 0 or sh <= 0:
            return None  # can't validate without screen; don't block

        def in_bounds(x: float, y: float) -> bool:
            return 0 <= x < sw and 0 <= y < sh

        for idx, step in enumerate(plan.get("steps", [])):
            t = step.get("type")
            if t == "click":
                target = (step.get("target") or {})
                rect = target.get("rect")
                point = target.get("point")
                if rect:
                    x, y, w, h = rect.get("x"), rect.get("y"), rect.get("w"), rect.get("h")
                    if any(not isinstance(v, (int, float)) for v in (x, y, w, h)):
                        return f"step[{idx}] CLICK rect must have numeric x,y,w,h"
                    if w < 2 or h < 2:
                        return f"step[{idx}] CLICK rect width/height must be >= 2 (got w={w}, h={h})"
                    cx, cy = x + w / 2.0, y + h / 2.0
                    if not (in_bounds(x, y) and in_bounds(x + w - 1, y + h - 1) and in_bounds(cx, cy)):
                        return f"step[{idx}] CLICK rect must be fully within screen {sw}x{sh}"
                elif point:
                    px, py = point.get("x"), point.get("y")
                    if any(not isinstance(v, (int, float)) for v in (px, py)):
                        return f"step[{idx}] CLICK point must have numeric x,y"
                    if not in_bounds(px, py):
                        return f"step[{idx}] CLICK point must be within screen {sw}x{sh}"
                else:
                    return f"step[{idx}] CLICK target must include either rect or point"
                cc = step.get("click_count", 1)
                if not isinstance(cc, int) or cc < 1 or cc > 2:
                    return f"step[{idx}] CLICK click_count must be 1 or 2"
            elif t == "drag":
                p1, p2 = step.get("from") or {}, step.get("to") or {}
                if any(not isinstance(p1.get(k), (int, float)) for k in ("x", "y")) or any(not isinstance(p2.get(k), (int, float)) for k in ("x", "y")):
                    return f"step[{idx}] DRAG must have numeric from/to"
                if not (in_bounds(p1["x"], p1["y"]) and in_bounds(p2["x"], p2["y"])):
                    return f"step[{idx}] DRAG points must be within screen {sw}x{sh}"
            elif t == "scroll":
                at = step.get("at")
                if at:
                    if any(not isinstance(at.get(k), (int, float)) for k in ("x", "y")):
                        return f"step[{idx}] SCROLL.at must have numeric x,y"
                    if not in_bounds(at["x"], at["y"]):
                        return f"step[{idx}] SCROLL.at must be within screen {sw}x{sh}"
            elif t == "move":
                pt = step.get("point") or {}
                if any(not isinstance(pt.get(k), (int, float)) for k in ("x", "y")):
                    return f"step[{idx}] MOVE.point must have numeric x,y"
                if not in_bounds(pt["x"], pt["y"]):
                    return f"step[{idx}] MOVE.point must be within screen {sw}x{sh}"
            elif t in ("type", "key_combo", "wait"):
                pass
            else:
                return f"step[{idx}] unknown action type: {t}"
        return None

    # ---- EXPECTED RESULT CHECK ----
    def _expected_holds(self, state: Dict[str, Any], expected_result: str) -> bool:
        text = (expected_result or "").lower()
        elems = (state or {}).get("elements", [])
        names = " ".join([(e.get("name") or "").lower() for e in elems])
        roles = " ".join([(e.get("role") or "").lower() for e in elems])

        # Coarse heuristics; adjust as you observe your app
        if "menubar is opened" in text:
            return ("menuitem" in roles) or ("popup" in roles) or ("menu" in roles)
        if "scenario management is shown" in text:
            return ("scenario" in names and ("open" in names or "management" in names))
        return False

    # -------------------- JSON repair helpers --------------------
    def _attempt_repair_plan(self, payload: str) -> Optional[str]:
        last_brace = payload.rfind("}")
        if last_brace == -1:
            return None
        candidate = payload[: last_brace + 1]
        attempt = self._close_unbalanced_delimiters(candidate)
        for _ in range(4):
            try:
                json.loads(attempt)
                return attempt
            except json.JSONDecodeError:
                trimmed = self._remove_trailing_comma(attempt)
                if trimmed == attempt:
                    break
                attempt = trimmed
        return None

    def _close_unbalanced_delimiters(self, text: str) -> str:
        in_string = False
        escape = False
        stack: List[str] = []
        for ch in text:
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
        repaired = text
        if escape:
            repaired += '\\'
        if in_string:
            repaired += '"'
        while stack:
            opener = stack.pop()
            repaired += '}' if opener == '{' else ']'
        return repaired

    def _remove_trailing_comma(self, text: str) -> str:
        idx = len(text) - 1
        while idx >= 0 and text[idx].isspace():
            idx -= 1
        while idx >= 0 and text[idx] in "]}":
            idx -= 1
            while idx >= 0 and text[idx].isspace():
                idx -= 1
        if idx >= 0 and text[idx] == ',':
            return text[:idx] + text[idx + 1:]
        return text

    def _prune_state_blobs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pruned: Dict[str, Any] = {}
        for key, value in state.items():
            if key in self._SCREENSHOT_KEYS and isinstance(value, str):
                pruned[key] = f"<omitted {len(value)} chars>"
                continue
            if isinstance(value, dict):
                pruned[key] = self._prune_state_blobs(value)
            else:
                pruned[key] = value
        return pruned


__all__ = [
    "LLMBackend",
    "StepDefinition",
    "ScenarioResult",
    "ScenarioStepOutcome",
    "RunResult",
    "ActionExecutionLog",
    "BackendError",
    "PlanParseError",
    "SUTCommunicationError",
    "LLMCommunicationError",
    "AGEN_TEST_PLAN_SCHEMA",
]
