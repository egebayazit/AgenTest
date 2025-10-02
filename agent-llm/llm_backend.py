# llm_backend.py
# Core LLM orchestration for Agentest
#
# Responsibilities:
# - Fetch state from SUT
# - Build an LLM prompt with SUT state + tester's step
# - Call OpenRouter to get an action plan (strict JSON)
# - Execute the plan on SUT
# - Fetch new state and run a simple verification
#
# Notes:
# - Kept intentionally framework-agnostic (no FastAPI/Streamlit here)
# - Robust content extraction for OpenRouter providers
# - Defensive JSON parsing (handles ```json fences etc.)
# - Reasonable timeouts and helpful logging

from __future__ import annotations

import json
import re
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentest.llm_backend")

# ---------- Data Models ----------

@dataclass
class TestStep:
    """A single test step coming from the UI."""
    step: str
    expected_result: str
    note_to_llm: str = ""


@dataclass
class Config:
    """Runtime configuration for the LLM backend."""
    openrouter_api_key: str
    sut_state_url: str = "http://127.0.0.1:18080/state"
    sut_action_url: str = "http://127.0.0.1:18080/action"
    model: str = "openai/gpt-oss-20b:free"
    timeout_connect: int = 10
    timeout_read: int = 60
    max_tokens: int = 1024
    temperature: float = 0.1


# ---------- Backend ----------

class LLMBackend:
    """
    Main LLM orchestration component:
    state -> prompt -> plan -> action -> verify
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: Config):
        self.config = config
        self.system_prompt = self._build_system_prompt()
        self.session = requests.Session()

    # ----- Prompt spec -----

    def _build_system_prompt(self) -> str:
        return (
        "ROLE: You are an expert Windows UI automation PLANNER.\n"
        "INPUTS: (a) test step, (b) expected result, (c) CURRENT UI STATE (elements with rects).\n"
        "TASK: Produce a VALID JSON action plan using ONLY elements present in the state OR keyboard shortcuts when no element matches.\n"
        "HARD RULES:\n"
        "- Return ONE JSON OBJECT only. No markdown fences, no commentary, no role tags.\n"
        "- Do NOT invent coordinates; always use element rects from state when clicking.\n"
        "- Prefer enabled + active-window elements. If no UI element matches, you MUST use keyboard-only steps.\n"
        "OUTPUT SCHEMA (exact keys):\n"
        "{\n"
        '  "action_id": "step_<timestamp>",\n'
        '  "coords_space": "physical",\n'
        '  "steps": [ /* array of actions */ ],\n'
        '  "reasoning": "short explanation"\n'
        "}\n"
        "ACTIONS (allowed):\n"
        '- CLICK: {"type":"click","button":"left|right|middle","click_count":1,'
        '"modifiers":["ctrl","shift","alt","win"],'
        '"target":{"rect":{"x":100,"y":100,"w":80,"h":30}}}\n'
        '- TYPE: {"type":"type","text":"...","delay_ms":10,"enter":false}  // set enter=true to press Enter after typing\n'
        '- KEY_COMBO: {"type":"key_combo","combo":["win","s"]}\n'
        "  Allowed keys in combo: ctrl, shift, alt, win, enter, tab, esc, backspace, delete, home, end,\n"
        "  pageup, pagedown, up, down, left, right, f1..f12, a..z, 0..9.\n"
        '- DRAG: {"type":"drag","from":{"x":..,"y":..},"to":{"x":..,"y":..},"button":"left","hold_ms":120}\n'
        '- SCROLL: {"type":"scroll","delta":-240,"horizontal":false,"at":{"x":800,"y":600}}\n'
        '- MOVE: {"type":"move","point":{"x":..,"y":..},"settle_ms":150}\n'
        '- WAIT: {"type":"wait","ms":300}\n'
        "SELECTION RULES:\n"
        "1) Search all elements; prefer enabled + visible ones.\n"
        "2) Use rect CENTER for clicks; never guess coordinates.\n"
        "3) If NO suitable element exists for the step, produce a KEYBOARD-ONLY plan.\n"
        "EXAMPLE (no Start element in state, open Settings via search):\n"
        "{\n"
        '  "action_id":"step_<timestamp>",\n'
        '  "coords_space":"physical",\n'
        '  "steps":[\n'
        '    {"type":"key_combo","combo":["win","s"]},\n'
        '    {"type":"wait","ms":250},\n'
        '    {"type":"type","text":"Settings","delay_ms":10,"enter":true}\n'
        '  ],\n'
        '  "reasoning":"No Start button element; using Win+S search, typing Settings, confirming with Enter."\n'
        "}\n"
        "Return ONLY the JSON object."
    )

    # ----- SUT I/O -----

    def get_state_from_controller(self, hint: str | None = None) -> Optional[Dict[str, Any]]:
        """Retrieve current UI state from Controller-SUT (/state)."""
        try:
            payload = {"hint": hint} if hint else {}
            r = self.session.post(
                self.config.sut_state_url,
                json=payload,                       # <— boş {} yerine payload
                timeout=(self.config.timeout_connect, self.config.timeout_read),
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("SUT /state failed: %s", e)
            return None

    def execute_action(self, action_plan: Dict[str, Any]) -> bool:
        """Send the action plan to SUT (/action). Expects ACK-style response."""
        try:
            r = self.session.post(
                self.config.sut_action_url,
                json=action_plan,
                timeout=(self.config.timeout_connect, self.config.timeout_read),
            )
            r.raise_for_status()
            body = r.json()
            if body.get("status") == "ok":
                logger.info("SUT /action ok; applied=%s", body.get("applied"))
                return True
            logger.error("SUT /action error: %s", body)
            return False
        except Exception as e:
            logger.error("SUT /action failed: %s", e)
            return False

    # ----- OpenRouter -----

    @staticmethod
    def _extract_content(result: Dict[str, Any]) -> str:
        """
        Providers may return content in various shapes:
        - choices[0].message.content -> str or list[parts{text}]
        - choices[0].text
        - choices[0].content (rare)
        If content is empty, try reasoning or refusal fields for diagnostics.
        """
        choices = result.get("choices") or []
        if not choices:
            return ""
        ch0 = choices[0]
        msg = ch0.get("message", {})
        c = msg.get("content", "")
        if isinstance(c, str) and c:
            return c
        if isinstance(c, list):
            parts: List[str] = []
            for p in c:
                if isinstance(p, dict):
                    if isinstance(p.get("text"), str):
                        parts.append(p["text"])
                    elif isinstance(p.get("content"), str):
                        parts.append(p["content"])
            if parts:
                return "".join(parts)
        # completion format
        t = ch0.get("text")
        if isinstance(t, str) and t:
            return t
        # very rare fallbacks
        c2 = ch0.get("content")
        if isinstance(c2, str) and c2:
            return c2
        # Eğer content boşsa, reasoning veya refusal varsa onları döndür
        reasoning = msg.get("reasoning")
        refusal = msg.get("refusal")
        if reasoning:
            logger.warning(f"LLM reasoning (no content): {reasoning}")
            return f"LLM reasoning: {reasoning}"
        if refusal:
            logger.warning(f"LLM refusal: {refusal}")
            return f"LLM refusal: {refusal}"
        return ""

    @staticmethod
    def _strip_code_fence(s: str) -> str:
        """Remove ```json ... ``` fences if present."""
        s = s.strip()
        if s.startswith("```"):
            # drop first ```
            s = s.split("```", 1)[1]
            # remove leading 'json' if provided
            s = s[4:] if s.startswith("json") else s
            # drop trailing ```
            s = s.rsplit("```", 1)[0]
        return s.strip()

    @staticmethod
    def _best_effort_json(s: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse JSON; if it fails, extract the first {...} block.
        """
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    def _call_openrouter(self, messages: List[Dict[str, Any]]) -> str:
        """Call OpenRouter and return raw content string."""
        if not self.config.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing.")

        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Agentest LLM Backend",
            "Accept": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            # Ask for structured JSON; some free models may ignore it, we still parse defensively.
            "response_format": {"type": "json_object"},
        }

        try:
            r = self.session.post(
                self.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=(self.config.timeout_connect, self.config.timeout_read),
            )
            r.raise_for_status()
        except requests.exceptions.ReadTimeout:
            raise RuntimeError(f"LLM read timeout (>{self.config.timeout_read}s)")
        except requests.exceptions.ConnectTimeout:
            raise RuntimeError(f"Connection timeout (>{self.config.timeout_connect}s)")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP {r.status_code} from OpenRouter: {r.text[:600]}") from e

        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"Non-JSON response from OpenRouter: {r.text[:600]}")

        content = self._extract_content(data)
        if not content:
            # Surface payload for diagnosis
            raise RuntimeError(f"Unexpected OpenRouter payload: {json.dumps(data)[:800]}")
        return content

    # ----- Orchestration -----

    def prepare_llm_context(self, test_step: TestStep, state: Dict[str, Any]) -> str:
        """
        Create a compact, LLM-friendly context that lists ALL named UI elements
        and screen metadata, plus the tester's step/expected/note.
        """
        elements = state.get("elements", [])
        screen = state.get("screen", {})

        simplified = []
        for e in elements:
            name = e.get("name")
            if not name:
                continue
            simplified.append({
                "idx": e.get("idx"),
                "name": name[:150],
                "type": e.get("controlType", ""),
                "rect": e.get("rect", {}),
                "enabled": e.get("enabled", False),
                "windowActive": e.get("windowActive", False),
                "path": (e.get("path") or [])[-2:],  # last two parents for context
            })

        ctx = (
            f"TEST STEP: {test_step.step}\n\n"
            f"EXPECTED RESULT: {test_step.expected_result}\n\n"
            + (f"NOTE TO LLM: {test_step.note_to_llm}\n\n" if test_step.note_to_llm else "")
            + "SCREEN INFO:\n"
            f"Resolution: {screen.get('w', 0)}x{screen.get('h', 0)}\n"
            f"DPI: {screen.get('dpiX', 96)}\n\n"
            f"AVAILABLE UI ELEMENTS (Total: {len(simplified)}):\n"
            f"{json.dumps(simplified, indent=2)}\n\n"
            "INSTRUCTIONS:\n"
            "1) Search through ALL elements above and pick the best match.\n"
            "2) Use the element rect to compute click coordinates.\n"
            "3) Prefer enabled + active-window elements.\n"
            "4) Return a VALID JSON action plan as per the schema (no markdown).\n"
        )
        return ctx

    def call_llm(self, test_step: TestStep, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:        
        """Get an action plan JSON from the LLM."""
        try:
            context = self.prepare_llm_context(test_step, state)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "text", "text": context}]},
            ]
            raw = self._call_openrouter(messages)

            # Reasoning / refusal fallback
            if raw.startswith("LLM reasoning:") or raw.startswith("LLM refusal:"):
                logger.error(f"LLM did not return content. {raw}")
                return None

            raw = self._strip_code_fence(raw)
            plan = self._best_effort_json(raw)
            if not plan or not isinstance(plan, dict):
                logger.error("Failed to parse LLM JSON. Raw:\n%s", raw[:800])
                return None

            # burada sadece hafif default ver, sanitizasyon yok
            plan.setdefault("action_id", f"step_{int(time.time())}")
            plan.setdefault("coords_space", "physical")
            plan.setdefault("steps", [])
            return plan
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def verify_result(
        self,
        test_step: TestStep,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Very simple heuristic verifier:
        - Split expected_result into tokens
        - Check if any token appears in element names of the new state
        """
        elems_after = state_after.get("elements", [])
        names = [str(e.get("name", "")).lower() for e in elems_after]
        tokens = [t for t in test_step.expected_result.lower().split() if t]
        hits = sum(1 for tok in tokens if any(tok in n for n in names))
        passed = (hits > 0) or (len(tokens) == 0)
        return {
            "passed": passed,
            "confidence": "low",
            "details": f"Matched {hits}/{len(tokens)} expected keywords in UI element names.",
            "state_before_elements": len(state_before.get("elements", [])),
            "state_after_elements": len(elems_after),
        }

    def run_test_step(self, test_step: TestStep) -> Dict[str, Any]:
        """
        One-shot flow for a single step:
        1) Get state
        2) Get plan from LLM
        3) Execute plan
        4) Get new state and verify
        """
        result: Dict[str, Any] = {
            "success": False,
            "step": test_step.step,
            "expected": test_step.expected_result,
            "actual": "",
            "passed": False,
            "logs": [],
        }

        def log_line(s: str):
            line = f"[{time.strftime('%H:%M:%S')}] {s}"
            result["logs"].append(line)
            logger.info(s)

        log_line("Starting test execution")
        log_line("Step 1: Fetching SUT state")
        before = self.get_state_from_controller()
        if not before:
            result["actual"] = "Failed to fetch SUT state"
            return result

        log_line("Step 2: Asking LLM for action plan")
        plan = self.call_llm(test_step, before)
        if not plan or not plan.get("steps"):
            result["actual"] = "LLM did not return a valid plan"
            return result
        result["action_plan"] = plan

        log_line(f"Step 3: Executing {len(plan['steps'])} step(s) on SUT")
        if not self.execute_action(plan):
            result["actual"] = "SUT failed to execute the plan"
            return result

        log_line("Step 4: Waiting for UI to settle")
        time.sleep(0.5)

        log_line("Step 5: Fetching SUT state again and verifying")
        after = self.get_state_from_controller()
        if not after:
            result["actual"] = "Failed to fetch SUT state after action"
            return result

        ver = self.verify_result(test_step, before, after)
        result["verification"] = ver
        result["passed"] = bool(ver.get("passed"))
        result["success"] = True
        result["actual"] = ver.get("details", "")
        log_line(f"Completed: {'PASSED' if result['passed'] else 'FAILED'}")
        return result