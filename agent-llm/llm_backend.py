# llm_backend.py
#LLM orchestration for AgenTest

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from difflib import SequenceMatcher
import httpx

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPT 
# ============================================================================

SYSTEM_PROMPT = """
system:
  role: AgenTest
  description: Expert Windows UI automation action planner
  output: VALID JSON ONLY (no markdown, no extra text)

mission: Convert ONE manual test step into ONE precise executable UI action.

CRITICAL_RULES[5]:
  1, ONE_STEP_ONE_ACTION: Each test step = EXACTLY 1 primary action (click/type)
  2, NO_WAIT_SPAM: Use wait ONLY if test step explicitly mentions "wait for..."
  3, NO_RANDOM_CLICKS: If element not found, return steps:[] with clear reasoning
  4, NO_RETRY_LOGIC: You attempt once, backend handles retries
  5, TRUST_COORDINATES: Given coordinates are ALWAYS accurate, never modify

json_format{key,type,desc}:
  action_id,string,Unique step ID
  coords_space,string,"physical"
  steps,array,List of actions (empty [] if not found)
  reasoning,string,max 100 words, no reference to system/system_prompt/TOON

input_data[7]:
  test_step,expected_result,current_state,spatial_analysis,recent_actions,screen,retry_context

element{field,desc}:
  name,WinDriver_text
  name_ocr,OCR_text
  name_ods,ODS_text
  center(x,y),Exact_coordinates

data_quality_order[3]: coordinates,name,ocr_ods

supported_actions[6]:
  click(type,button,click_count,modifiers,target_point)
  type(text,delay_ms,enter)
  key_combo(combo_keys)
  wait(ms)
  drag(from,to,button,hold_ms)
  scroll(delta,horizontal,at)

wait_constraints:
  require_explicit_mention_in_test_step: true
  no_automatic_waits: true

matching_priority[5]:
  1,spatial_analysis
  2,exact_name_match
  3,partial_match_ods_ocr
  4,row_based_selection
  5,empty_if_not_found

spatial_analysis_rules:
  use_first_candidate_only: true
  use_exact_center: true
  ignore_all_other_matching: true

exact_name_rules:
  match_field: name
  case_insensitive: true

partial_matching:
  sources[2]: name_ods,name_ocr
  requires_coordinate_sense: true
  min_confidence: 0.8

row_based_selection:
  conditions[3]: label_exists, same_row(|y_label - y| ≤ 10), control_is_right_side
  allowed_generic_names: "",Image,Checkmark,formViewField,BoolAttributeView
  strategy: choose_nearest_x_distance
  note: generic_elements_only_clickable_in_row_based_selection

screen_constraints:
  enforce_bounds: true
  invalid_if_outside_screen: true

not_found_behavior:
  steps: []
  reasoning: "Element not found with confidence > 80%"

absolute_rules[12]:
  1, use_exact_coordinates_only
  2, never_generate_coordinates
  3, never_modify_given_coordinates
  4, never_use_coordinates_outside_screen_bounds
  5, never_repeat_recent_actions
  6, no_keyboard_fallback_when_missing_element
  7, never_infer_from_expected_result
  8, never_click_center_of_screen
  9, never_invent_elements
 10, never_guess_spatial_relations
 11, never_click_generic_elements_outside_row_based_selection
 12, return_empty_steps_if_unsure

retry_context:
  attempt1: WinDriver (strict, exact only)
  attempt2: ODS (enhanced, partial + row-based allowed)

notes[7]:
  coordinates_always_accurate
  spatial_analysis_highest_priority
  ocr_ods_text_may_be_incomplete
  generic_elements_often_true_controls
  focus_only_on_current_test_step
  reasoning_cannot_reference_rules_system_or_prompt
  precision_over_creativity

example_not_found_json:
  action_id: step_x
  coords_space: physical
  steps: []
  reasoning: "Target element not found"

reminder:
  output_must_be_valid_json_only
  no_text_before_or_after_json
  task_is_planning_exact_physical_interactions
"""

# ============================================================================
# JSON SCHEMA
# ============================================================================

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
            "maxItems": 3,  # ✅ HARD LIMIT: Max 3 steps   #todo: check limits
            "items": {
                "type": "object",
                "additionalProperties": False,
                "oneOf": [
                    {  # CLICK with point
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
                                "required": ["point"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["type", "button", "click_count", "target"],
                    },
                    {  # TYPE
                        "properties": {
                            "type": {"const": "type"},
                            "text": {"type": "string"},
                            "delay_ms": {"type": "integer", "minimum": 0},
                            "enter": {"type": "boolean"},
                        },
                        "required": ["type", "text"],
                    },
                    {  # KEY_COMBO
                        "properties": {
                            "type": {"const": "key_combo"},
                            "combo": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                        "required": ["type", "combo"],
                        "additionalProperties": False,
                    },
                    {  # WAIT
                        "properties": {
                            "type": {"const": "wait"},
                            "ms": {"type": "integer", "minimum": 0},
                        },
                        "required": ["type", "ms"],
                    },
                    {  # DRAG
                        "properties": {
                            "type": {"const": "drag"},
                            "from": {
                                "type": "object",
                                "required": ["x", "y"],
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                            },
                            "to": {
                                "type": "object",
                                "required": ["x", "y"],
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                            },
                            "button": {"enum": ["left", "right", "middle"]},
                            "hold_ms": {"type": "integer", "minimum": 0},
                        },
                        "required": ["type", "from", "to"],
                    },
                    {  # SCROLL
                        "properties": {
                            "type": {"const": "scroll"},
                            "delta": {"type": "integer"},
                            "horizontal": {"type": "boolean"},
                            "at": {
                                "type": "object",
                                "required": ["x", "y"],
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                            },
                        },
                        "required": ["type", "delta"],
                    },
                ],
            },
        },
    },
}

# ============================================================================
# EXCEPTIONS
# ============================================================================

class BackendError(Exception):
    """Base exception for backend errors"""
    pass


class PlanParseError(BackendError):
    """LLM returned invalid or unparseable plan"""
    pass


class SUTCommunicationError(BackendError):
    """Failed to communicate with SUT"""
    pass


class LLMCommunicationError(BackendError):
    """Failed to communicate with LLM provider"""
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StepDefinition:
    """A single test step"""
    test_step: str
    expected_result: str
    note_to_llm: Optional[str] = None


@dataclass
class ActionExecutionLog:
    """Log of a single action execution"""
    action_id: str
    plan: Dict[str, Any]
    ack: Dict[str, Any]
    state_before: Dict[str, Any]
    state_after: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class RunResult:
    """Result of running a single test step"""
    status: str  # "passed", "failed", "error"
    attempts: int
    actions: List[ActionExecutionLog]
    final_state: Optional[Dict[str, Any]]
    last_plan: Optional[Dict[str, Any]]
    reason: Optional[str] = None


@dataclass
class ScenarioStepOutcome:
    """Outcome of a single scenario step"""
    step: StepDefinition
    result: RunResult


@dataclass
class ScenarioResult:
    """Result of running a full scenario"""
    status: str  # "passed", "failed", "error"
    steps: List[ScenarioStepOutcome]
    final_state: Optional[Dict[str, Any]]
    reason: Optional[str] = None


# ============================================================================
# LLM BACKEND - WITH ENHANCED VALIDATION
# ============================================================================

class LLMBackend:
    """LLM-based action planner backend with enhanced validation"""
    
    _SCREENSHOT_KEYS = {"screenshot", "screenshot_png", "screenshot_jpg", "screenshot_base64", "b64"}

    # ========================================================================
    # SPATIAL PATTERN DEFINITIONS
    # ========================================================================
    
    _SPATIAL_PATTERNS = {
        "right": [
            r'(?:dropdown|button|field|checkbox|menu|element|control)\s+(?:on the right of|to the right of|right of|at the right side of)\s+(.+?)(?:\s*$|,)',
            r'(?:on the right of|to the right of|right of|at the right side of)\s+(.+?)(?:\s*$|,)',
        ],
        "left": [
            r'(?:dropdown|button|field|checkbox|menu|element|control)\s+(?:on the left of|to the left of|left of|at the left side of)\s+(.+?)(?:\s*$|,)',
            r'(?:on the left of|to the left of|left of|at the left side of)\s+(.+?)(?:\s*$|,)',
        ],
        "near": [ 
            r'(?:dropdown|button|field|checkbox|menu|element|control)\s+(?:next to|beside|near|close to|around|nearby|adjacent to)\s+(.+?)(?:\s*$|,)',
            r'(?:next to|beside|near|close to|around|nearby|adjacent to)\s+(.+?)(?:\s*$|,)',
        ],
        "above": [
            r'(?:dropdown|button|field|checkbox|menu|element|control|label)\s+(?:above|over|top of|upper)\s+(.+?)(?:\s*$|,)',
            r'(?:above|over|top of|upper)\s+(.+?)(?:\s*$|,)',
        ],
        "below": [
            r'(?:dropdown|button|field|checkbox|menu|element|control)\s+(?:below|under|beneath|bottom of|lower)\s+(.+?)(?:\s*$|,)',
            r'(?:below|under|beneath|bottom of|lower)\s+(.+?)(?:\s*$|,)',
        ],
        "same_row": [
            r'same row as\s+(.+?)(?:\s*$|,)',
            r'in the\s+(.+?)\s+row',
        ],
        "same_column": [
            r'same column as\s+(.+?)(?:\s*$|,)',
            r'in the\s+(.+?)\s+column',
        ],
    }

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
        system_prompt: str = SYSTEM_PROMPT,
        max_attempts: int = 2, 
        post_action_delay: float = 0.5,
        sut_timeout: float = 50.0,
        llm_timeout: float = 800.0,
        max_tokens: int = 384,  # 600 → 384 
        max_plan_steps: int = 10,  # 24 → 10 (spam action önleme için)
        schema_retry_limit: int = 1,
        http_referrer: str = "https://agentest.local/backend",
        client_title: str = "AgenTest LLM Backend",
        omit_large_blobs: bool = True,
        enforce_json_response: bool = True,
    ) -> None:

        # Validation
        if not llm_model:
            raise ValueError("llm_model is required")
        
        if llm_provider not in ("ollama", "openrouter", "lmstudio"):
            raise ValueError("llm_provider must be 'ollama', 'lmstudio'or 'oropenrouter'")
        
        if llm_provider == "openrouter" and not llm_api_key:
            raise ValueError("llm_api_key required for OpenRouter")
        
        if max_plan_steps <= 0:
            raise ValueError("max_plan_steps must be positive")
        if schema_retry_limit < 0:
            raise ValueError("schema_retry_limit must be >= 0")

        # Store config
        self.state_url_windriver = state_url_windriver
        self.state_url_ods = state_url_ods
        self.max_attempts = max_attempts
        
        # LLM config
        self.llm_provider = llm_provider
        self.model = llm_model
        self.llm_base_url = llm_base_url.rstrip("/")
        self.api_key = llm_api_key
        
        # Other config
        self.enforce_json_response = enforce_json_response
        self._json_response_enabled = enforce_json_response
        self.action_url = action_url
        self.system_prompt = system_prompt.strip()
        self.post_action_delay = post_action_delay
        self.sut_timeout = httpx.Timeout(sut_timeout)
        self.llm_timeout = httpx.Timeout(llm_timeout)
        self.max_tokens = max_tokens
        self.max_plan_steps = max_plan_steps
        self.schema_retry_limit = schema_retry_limit
        self.http_referrer = http_referrer
        self.client_title = client_title
        self.omit_large_blobs = omit_large_blobs
                
        self._last_element_count = 0
        self._previous_state_hash: Optional[str] = None

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMBackend":
        """Create backend from environment variables"""
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
            "state_url_windriver": overrides.pop(
                "state_url_windriver", 
                env("SUT_STATE_URL_WINDRIVER", "http://127.0.0.1:18800/state/for-llm")
            ),
            "state_url_ods": overrides.pop(
                "state_url_ods", 
                env("SUT_STATE_URL_ODS", "http://127.0.0.1:18800/state/from-ods")
            ),
            "action_url": overrides.pop(
                "action_url", 
                env("SUT_ACTION_URL", "http://192.168.137.249:18080/action")
            ),
            "llm_provider": overrides.pop(
                "llm_provider",
                env("LLM_PROVIDER", "ollama")
            ),
            "llm_model": overrides.pop(
                "llm_model",
                env("LLM_MODEL", "mistral-small3.2:latest")
            ),
            "llm_base_url": overrides.pop(
                "llm_base_url",
                env("LLM_BASE_URL", "http://localhost:11434")
            ),
            "llm_api_key": overrides.pop(
                "llm_api_key",
                env("LLM_API_KEY") or env("OPENROUTER_API_KEY", None)
            ),
            "enforce_json_response": _env_bool("ENFORCE_JSON_RESPONSE"),
        }
        
        params.update(overrides)
        return cls(**params)

    # ========================================================================
    # SPATIAL ANALYSIS
    # ========================================================================
    
    def _generate_spatial_hints(
        self, 
        test_step: str, 
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate spatial relationship hints"""
        detected_direction = None
        reference_label = None
        
        logger.debug("  → Analyzing test_step for spatial patterns: '%s'", test_step)
        
        for direction, patterns in self._SPATIAL_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, test_step, re.IGNORECASE)
                if match:
                    detected_direction = direction
                    reference_label = match.group(1).strip()
                    logger.info("  → Detected spatial: %s of '%s'", direction, reference_label)
                    break
            if detected_direction:
                break
        
        if not detected_direction or not reference_label:
            logger.debug("  → No spatial relationship detected in test_step")
            return None
        
        elements = state.get("elements", [])
        ref_elem = None
        
        for elem in elements:
            name = elem.get("name", "").strip()
            if name.lower() == reference_label.lower():
                ref_elem = elem
                logger.debug("  → Exact match: '%s'", name)
                break
        
        if not ref_elem:
            for elem in elements:
                name = elem.get("name", "").strip()
                if reference_label.lower() in name.lower():
                    ref_elem = elem
                    logger.debug("  → Partial match: '%s'", name)
                    break
        
        if not ref_elem:
            best_match = None
            best_ratio = 0.0
            
            for elem in elements:
                name = elem.get("name", "").strip()
                if not name:
                    continue
                
                ratio = SequenceMatcher(None, reference_label.lower(), name.lower()).ratio()
                if ratio >= 0.70 and ratio > best_ratio:
                    best_ratio = ratio
                    best_match = elem
            
            if best_match:
                ref_elem = best_match
                logger.debug("  → Fuzzy match: '%s' (%.1f%% similarity)", 
                           best_match.get("name", ""), best_ratio * 100)
        
        if not ref_elem:
            logger.debug("  → Spatial hint: Reference label '%s' not found", reference_label)
            return None
        
        ref_x = ref_elem["center"]["x"]
        ref_y = ref_elem["center"]["y"]
        
        logger.debug("  → Found reference '%s' at (%d, %d)", reference_label, ref_x, ref_y)
        
        nearby_elements = self._search_by_direction(
            elements=elements,
            direction=detected_direction,
            ref_x=ref_x,
            ref_y=ref_y,
            ref_elem=ref_elem
        )
        
        if not nearby_elements:
            logger.debug("  → No elements found %s of '%s'", detected_direction, reference_label)
            return None
        
        nearby_elements.sort(key=lambda e: e["distance"])
        
        logger.debug(
            "  → Spatial hint: Found %d elements %s of '%s'",
            len(nearby_elements),
            detected_direction,
            reference_label
        )
        
        hint_text = (
            f"Found '{reference_label}' at ({ref_x}, {ref_y}). "
            f"{len(nearby_elements)} elements detected {detected_direction}. "
        )
        if nearby_elements:
            closest = nearby_elements[0]
            hint_text += (
                f"Closest: '{closest['name']}' at "
                f"({closest['center']['x']}, {closest['center']['y']})"
            )
        
        return {
            "reference_label": reference_label,
            "reference_location": {"x": ref_x, "y": ref_y},
            "spatial_direction": detected_direction,
            "nearby_candidates": nearby_elements[:3],  # 10 → 3 (context küçültme)
            "hint": hint_text
        }

    def _search_by_direction(
        self,
        elements: List[Dict[str, Any]],
        direction: str,
        ref_x: float,
        ref_y: float,
        ref_elem: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search elements in specific direction from reference point"""
        nearby = []
        
        for elem in elements:
            if elem is ref_elem:
                continue
            
            ex = elem["center"]["x"]
            ey = elem["center"]["y"]
            
            dx = ex - ref_x
            dy = ey - ref_y
            
            if direction == "right":
                if 0 < dx < 200 and abs(dy) < 40:
                    nearby.append({
                        "name": elem.get("name", ""),
                        "center": elem["center"],
                        "distance": abs(dx) + abs(dy),
                        "direction": "right",
                        "dx": dx,
                        "dy": dy,
                    })
            
            elif direction == "left":
                if -200 < dx < 0 and abs(dy) < 40:
                    nearby.append({
                        "name": elem.get("name", ""),
                        "center": elem["center"],
                        "distance": abs(dx) + abs(dy),
                        "direction": "left",
                        "dx": dx,
                        "dy": dy,
                    })
            
            elif direction == "above":
                if -120 < dy < 0 and abs(dx) < 80:
                    nearby.append({
                        "name": elem.get("name", ""),
                        "center": elem["center"],
                        "distance": abs(dx) + abs(dy),
                        "direction": "above",
                        "dx": dx,
                        "dy": dy,
                    })
            
            elif direction == "below":
                if 0 < dy < 120 and abs(dx) < 80:
                    nearby.append({
                        "name": elem.get("name", ""),
                        "center": elem["center"],
                        "distance": abs(dx) + abs(dy),
                        "direction": "below",
                        "dx": dx,
                        "dy": dy,
                    })
            
            elif direction == "near":
                distance = (dx**2 + dy**2) ** 0.5
                if distance < 200:
                    nearby.append({
                        "name": elem.get("name", ""),
                        "center": elem["center"],
                        "distance": distance,
                        "direction": "near",
                        "dx": dx,
                        "dy": dy,
                    })
            
            elif direction == "same_row":
                if abs(dy) < 15:
                    nearby.append({
                        "name": elem.get("name", ""),
                        "center": elem["center"],
                        "distance": abs(dx),
                        "direction": "same_row",
                        "dx": dx,
                        "dy": dy,
                    })
            
            elif direction == "same_column":
                if abs(dx) < 15:
                    nearby.append({
                        "name": elem.get("name", ""),
                        "center": elem["center"],
                        "distance": abs(dy),
                        "direction": "same_column",
                        "dx": dx,
                        "dy": dy,
                    })
        
        return nearby

    # ========================================================================
    # SCENARIO & STEP EXECUTION
    # ========================================================================

    async def run_scenario(
        self,
        steps: List[StepDefinition],
        *,
        temperature: float = 0.1,
    ) -> ScenarioResult:
        """Run a full test scenario (multiple steps)"""
        if not steps:
            raise ValueError("steps must contain at least one StepDefinition")

        history: List[Dict[str, Any]] = []
        outcomes: List[ScenarioStepOutcome] = []
        final_state: Optional[Dict[str, Any]] = None

        for step_index, step in enumerate(steps, 1):
            logger.info("=" * 80)
            logger.info("EXECUTING STEP %d/%d", step_index, len(steps))
            logger.info("=" * 80)
            
            result = await self.run_step(
                test_step=step.test_step,
                expected_result=step.expected_result,
                note_to_llm=step.note_to_llm,
                recent_actions=history,
                temperature=temperature,
            )
            
            outcomes.append(ScenarioStepOutcome(step=step, result=result))
            final_state = result.final_state
            
            for log in result.actions:
                history.append(self._summarise_for_prompt(log))
            if len(history) > 3:
                history = history[-3:]
            
            if result.status != "passed":
                return ScenarioResult(
                    status=result.status,
                    steps=outcomes,
                    final_state=final_state,
                    reason=result.reason
                )

        return ScenarioResult(status="passed", steps=outcomes, final_state=final_state)

    async def run_step(
        self,
        test_step: str,
        expected_result: str,
        note_to_llm: Optional[str] = None,
        *,
        recent_actions: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
    ) -> RunResult:
        """Run a single test step with enhanced validation"""
        history_payload: List[Dict[str, Any]] = list(recent_actions or [])
        actions_log: List[ActionExecutionLog] = []
        last_plan: Optional[Dict[str, Any]] = None
        state: Dict[str, Any] = {}

        for attempt in range(1, self.max_attempts + 1):
            detection_method = "WinDriver" if attempt == 1 else "ODS"
            logger.info("Attempt %d/%d (%s)", attempt, self.max_attempts, detection_method)
            
            state = await self._fetch_state(attempt_number=attempt)
            
            for schema_attempt in range(self.schema_retry_limit + 1):
                plan = await self._request_plan(
                    test_step=test_step,
                    expected_result=expected_result,
                    note_to_llm=note_to_llm,
                    state=state,
                    recent_actions=history_payload,
                    temperature=temperature,
                    schema_hint=None,
                    attempt_number=attempt,
                )

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
                    continue

                if plan.pop("_backend_steps_substituted", False):
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
                    continue

                break
            else:
                raise RuntimeError("Schema retry loop exited unexpectedly")

            last_plan = plan
            steps = plan.get("steps", [])
            reasoning = plan.get("reasoning", "")

            # STEP COUNT VALIDATION
            if len(steps) > self.max_plan_steps:
                logger.error("❌ LLM generated %d steps (max: %d) - REJECTING PLAN", 
                           len(steps), self.max_plan_steps)
                plan["steps"] = []
                plan["reasoning"] = f"Backend rejected plan: {len(steps)} steps exceeds limit of {self.max_plan_steps}"
                plan["_backend_steps_substituted"] = True
                steps = []
                reasoning = plan["reasoning"]

            logger.info("=" * 80)
            logger.info("📋 TEST STEP: %s", test_step)
            logger.info("🎯 EXPECTED: %s", expected_result[:100])
            logger.info("🤖 LLM PLAN: %d step(s)", len(steps))
            logger.info("💭 LLM REASONING: %s", reasoning[:150])
            
            if steps:
                logger.info("📍 ACTIONS TO EXECUTE:")
                for i, step in enumerate(steps, 1):
                    step_type = step.get("type", "unknown")
                    if step_type == "click":
                        point = step.get("target", {}).get("point", {})
                        button = step.get("button", "left")
                        count = step.get("click_count", 1)
                        logger.info("  %d. CLICK(%s, count=%d) at (%s, %s)", 
                                  i, button, count, 
                                  point.get('x', '?'), point.get('y', '?'))
                    elif step_type == "type":
                        text = step.get("text", "")[:30]
                        logger.info("  %d. TYPE '%s...'", i, text)
                    elif step_type == "wait":
                        ms = step.get("ms", 0)
                        logger.info("  %d. WAIT %dms", i, ms)
                    elif step_type == "key_combo":
                        combo = step.get("combo", [])
                        logger.info("  %d. KEY_COMBO %s", i, "+".join(combo))
                    else:
                        logger.info("  %d. %s", i, step_type.upper())
            else:
                logger.warning("  ⚠️  NO ACTIONS (empty steps)")
                logger.warning("  💭 Reason: %s", reasoning)
                
                if attempt < self.max_attempts:
                    logger.info("  → Retrying with ODS detection...")
                    continue
                else:
                    failure_message = f"Element not found after {self.max_attempts} attempts. LLM reasoning: {reasoning}"
                    logger.error("❌ FAILED: %s", failure_message)
                    
                    return RunResult(
                        status="failed",
                        attempts=attempt,
                        actions=actions_log,
                        final_state=state,
                        last_plan=plan,
                        reason=failure_message,
                    )
            
            logger.info("=" * 80)
            
            state_before = state

            ack = await self._send_action(plan)
            log_entry = ActionExecutionLog(
                action_id=plan.get("action_id", ""),
                plan=plan,
                ack=ack,
                state_before=state_before,
            )
            actions_log.append(log_entry)
            history_payload.append(self._summarise_for_prompt(log_entry))

            if ack.get("status") != "ok":
                final_state = await self._fetch_state_safe(state, attempt_number=attempt)
                log_entry.state_after = final_state
                message = f"SUT /action returned status '{ack.get('status')}': {ack.get('message', '')}"
                logger.error("✗ Execution error: %s", message)
                return RunResult(
                    status="error",
                    attempts=attempt,
                    actions=actions_log,
                    final_state=final_state,
                    last_plan=plan,
                    reason=message,
                )

            final_state = await self._fetch_state_safe(state, attempt_number=attempt)
            log_entry.state_after = final_state
            
            ui_changed, change_magnitude = self._detect_ui_change(state_before, final_state)
            
            logger.info("🔍 VALIDATING EXPECTED RESULT...")
            logger.info("  Expected: %s", expected_result[:100])
            logger.info("  UI changed: %s (magnitude: %.1f%%)", 
                       "YES" if ui_changed else "NO", 
                       change_magnitude * 100)
            
            validation_passed = self._expected_holds(
                final_state, 
                expected_result, 
                plan,
                state_before=state_before 
            )
            
            if validation_passed:
                visibility_keywords = ["visible", "appears", "shown", "displayed", "open", "shows", "loaded"]
                expects_visibility = any(kw in expected_result.lower() for kw in visibility_keywords)
                
                if expects_visibility and not ui_changed:
                    logger.warning("⚠️ Expected visibility but UI did not change!")
                    logger.warning("  Change magnitude: %.1f%%", change_magnitude * 100)
                    
                    if attempt >= self.max_attempts:
                        return RunResult(
                            status="failed",
                            attempts=attempt,
                            actions=actions_log,
                            final_state=final_state,
                            last_plan=plan,
                            reason="Expected UI visibility change but state remained same",
                        )
                    else:
                        logger.info("  → Retrying with ODS...")
                        continue
                
                logger.info("✓ SUCCESS: Expected result achieved")
                logger.info("  Validation: PASS ✓")
                return RunResult(
                    status="passed",
                    attempts=attempt,
                    actions=actions_log,
                    final_state=final_state,
                    last_plan=plan,
                    reason=f"Expected result achieved (detection: {detection_method})",
                )
            else:
                logger.info("  Validation: FAIL ✗")
                logger.info("  Expected result not yet achieved")
                
                if attempt >= self.max_attempts:
                    return RunResult(
                        status="failed",
                        attempts=attempt,
                        actions=actions_log,
                        final_state=final_state,
                        last_plan=plan,
                        reason=f"Expected result not achieved after {attempt} attempts (WinDriver + ODS)",
                    )
                
                logger.info("  → Retrying with ODS detection...")

        logger.warning("✗ Failed: Exhausted %d attempts", self.max_attempts)
        return RunResult(
            status="failed",
            attempts=self.max_attempts,
            actions=actions_log,
            final_state=final_state,
            last_plan=last_plan,
            reason=f"Exhausted {self.max_attempts} attempts (WinDriver + ODS) without achieving expected result.",
        )

    # ========================================================================
    # STATE & ACTION COMMUNICATION
    # ========================================================================

    async def _fetch_state(self, attempt_number: int = 1) -> Dict[str, Any]:
        """Fetch current state from SUT"""
        use_ods = (attempt_number == 2)
        state_url = self.state_url_ods if use_ods else self.state_url_windriver
        
        source_name = "ODS" if use_ods else "WinDriver"
        logger.info(
            "Fetching state (attempt %d/%s): %s",
            attempt_number,
            source_name,
            state_url
        )
        
        try:
            async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
                response = await client.post(state_url, json={})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            message = f"Failed to contact SUT /state at {state_url}: {exc}"
            logger.error(message)
            raise SUTCommunicationError(message) from exc

        try:
            state = response.json()
        except json.JSONDecodeError as exc:
            message = f"SUT /state returned invalid JSON from {source_name} endpoint."
            logger.error(message)
            raise SUTCommunicationError(message) from exc

        if self.omit_large_blobs and isinstance(state, dict):
            state = self._prune_state_blobs(state)
        
        element_count = len(state.get("elements", []))
        logger.debug("  → Received %d elements from %s endpoint", element_count, source_name)
        
        return state

    async def _fetch_state_safe(self, fallback: Dict[str, Any], attempt_number: int = 1) -> Dict[str, Any]:
        """Fetch state with fallback on error"""
        try:
            return await self._fetch_state(attempt_number)
        except BackendError:
            logger.warning("Failed to fetch state, using fallback")
            return fallback

    async def _send_action(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Send action plan to SUT for execution"""
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

    # ========================================================================
    # UI CHANGE DETECTION
    # ========================================================================
    
    def _compute_state_hash(self, state: Dict[str, Any]) -> str:
        """Compute hash of UI state for change detection"""
        elements = state.get("elements", [])
        
        state_signature = []
        for elem in elements:
            name = elem.get("name") or elem.get("name_ods") or elem.get("name_ocr") or ""
            center = elem.get("center", {})
            state_signature.append(f"{name}@{center.get('x', 0)},{center.get('y', 0)}")
        
        signature_str = "|".join(sorted(state_signature))
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def _detect_ui_change(
        self, 
        state_before: Dict[str, Any], 
        state_after: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Detect if UI changed significantly after action"""
        
        hash_before = self._compute_state_hash(state_before)
        hash_after = self._compute_state_hash(state_after)
        
        if hash_before == hash_after:
            return False, 0.0
        
        elems_before = len(state_before.get("elements", []))
        elems_after = len(state_after.get("elements", []))
        
        if elems_before == 0:
            change_ratio = 1.0 if elems_after > 0 else 0.0
        else:
            change_ratio = abs(elems_after - elems_before) / elems_before
        
        return True, change_ratio

    # ========================================================================
    # LLM COMMUNICATION 
    # ========================================================================

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
        attempt_number: int = 1,
    ) -> Dict[str, Any]:
        """Request action plan from LLM (Ollama or OpenRouter)"""
        messages = self._build_messages(
            test_step=test_step,
            expected_result=expected_result,
            note_to_llm=note_to_llm,
            state=state,
            recent_actions=recent_actions,
            schema_hint=schema_hint,
            attempt_number=attempt_number,
        )

        if self.llm_provider == "ollama":
            return await self._request_plan_ollama(messages, temperature)
        elif self.llm_provider == "lmstudio":
            return await self._request_plan_lmstudio(messages, temperature)
        else:
            return await self._request_plan_openrouter(messages, temperature)

    async def _request_plan_ollama(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        """Request plan from Ollama (local) with JSON Schema enforcement"""

        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": AGEN_TEST_PLAN_SCHEMA, 
            "options": {
                "temperature": temperature,
                "num_predict": self.max_tokens,
            },
        }
        
        url = f"{self.llm_base_url}/api/chat"
        
        logger.error("🚀 SENDING TO OLLAMA:")
        logger.error("URL: %s", url)
        logger.error("=" * 80)
        
        try:
            async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                response = await client.post(url, json=body)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            message = f"Ollama request failed: {exc}"
            logger.error(message)
            raise LLMCommunicationError(message) from exc
        
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            message = "Ollama returned invalid JSON"
            logger.error(message)
            raise LLMCommunicationError(message) from exc
        
        try:
            content = data["message"]["content"].strip()
        except (KeyError, AttributeError) as exc:
            message = f"Unexpected Ollama response format: {data}"
            logger.error(message)
            raise LLMCommunicationError(message) from exc
        
        logger.error("🔍 RAW LLM RESPONSE:")
        logger.error(content)
        logger.error("=" * 80)
    
        if not content:
            raise PlanParseError("Ollama returned empty content")
        
        plan = self._parse_plan(content)
        return plan
    
    async def _request_plan_lmstudio(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
) ->     Dict[str, Any]:
        """Request plan from LM Studio (OpenAI-compatible API)"""

        # LM Studio uses OpenAI-compatible /v1/chat/completions endpoint
        url = f"{self.llm_base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        # Add API key if provided (LM Studio doesn't require it by default)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        #  Add JSON schema if enforce_json_response is enabled
        if self.enforce_json_response:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "AgenTestPlan",
                    "schema": AGEN_TEST_PLAN_SCHEMA,
                },
            }

        logger.error("🚀 SENDING TO LM STUDIO:")
        logger.error("URL: %s", url)
        logger.error("Model: %s", self.model)
        logger.error("=" * 80)

        try:
            async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            message = f"LM Studio request failed: {exc}"
            logger.error(message)
            raise LLMCommunicationError(message) from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            message = "LM Studio returned invalid JSON"
            logger.error(message)
            raise LLMCommunicationError(message) from exc

        # Extract content from OpenAI-compatible response
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            message = f"Unexpected LM Studio response format: {data}"
            logger.error(message)
            raise LLMCommunicationError(message) from exc

        logger.error("🔍 RAW LLM RESPONSE (LM Studio):")
        logger.error(content)
        logger.error("=" * 80)

        if not content:
            raise PlanParseError("LM Studio returned empty content")

        plan = self._parse_plan(content)
        return plan

    async def _request_plan_openrouter(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        """Request plan from OpenRouter (fallback/cloud)"""
        OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
        
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
            body = self._build_request_body_openrouter(messages, temperature, enforce_json)
            
            try:
                async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                    response = await client.post(OPENROUTER_URL, headers=headers, json=body)
            except httpx.HTTPError as exc:
                message = f"OpenRouter request failed: {exc}"
                logger.error(message)
                raise LLMCommunicationError(message) from exc

            if response.status_code >= 400:
                response_text = response.text
                last_error = f"OpenRouter returned {response.status_code}: {response_text}"
                if enforce_json and ("JSON Schema Validation Error" in response_text or "response_format is not supported" in response_text):
                    logger.warning("Model rejected response_format; retrying without enforced JSON")
                    self._json_response_enabled = False
                    continue
                logger.error(last_error)
                raise LLMCommunicationError(last_error)

            data = response.json()
            
            try:
                content = self._extract_llm_content_openrouter(data)

                logger.error("🔍 RAW LLM RESPONSE (OpenRouter):")
                logger.error(content)
                logger.error("=" * 80) 

            except PlanParseError:
                if self.max_tokens > 400:
                    saved = self.max_tokens
                    self.max_tokens = 400
                    try:
                        body2 = self._build_request_body_openrouter(
                            messages, 
                            max(0.0, temperature - 0.05), 
                            self._json_response_enabled
                        )
                        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                            resp2 = await client.post(OPENROUTER_URL, headers=headers, json=body2)
                            resp2.raise_for_status()
                        data2 = resp2.json()
                        logger.warning("Retry with reduced tokens succeeded")
                        content = self._extract_llm_content_openrouter(data2)
                    finally:
                        self.max_tokens = saved
                else:
                    raise

            plan = self._parse_plan(content)
            self._json_response_enabled = bool(enforce_json)
            return plan

        if last_error:
            raise LLMCommunicationError(last_error)
        raise LLMCommunicationError("OpenRouter request failed without response")

    def _build_request_body_openrouter(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        enforce_json: bool
    ) -> Dict[str, Any]:
        """Build request body for OpenRouter API"""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "provider": {
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "stop": ["</JSON_ONLY>", "<|end|>", "assistant:", "commentary to=assistant"],
        }

        if enforce_json and self.enforce_json_response:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "AgenTestPlan",
                    "schema": AGEN_TEST_PLAN_SCHEMA,
                },
            }

        return body

    def _extract_llm_content_openrouter(self, data: Dict[str, Any]) -> str:
        """Extract content from OpenRouter response"""
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
            logger.warning("Using %s field as content fallback", fallback_source)
        
        return content

    def _build_messages(
        self,
        *,
        test_step: str,
        expected_result: str,
        note_to_llm: Optional[str],
        state: Dict[str, Any],
        recent_actions: List[Dict[str, Any]],
        schema_hint: Optional[str] = None,
        attempt_number: int = 1,
    ) -> List[Dict[str, Any]]:
        """Build messages for LLM request with spatial analysis"""
        screen_info = state.get("screen", {})
        
        payload: Dict[str, Any] = {
            "test_step": test_step,
            "expected_result": expected_result,
            "screen": {
                "w": screen_info.get("w"),
                "h": screen_info.get("h"),
            },
            "current_state": {
                "elements": state.get("elements", []),
            },
        }

        spatial_hints = self._generate_spatial_hints(test_step, state)
        if spatial_hints:
            payload["spatial_analysis"] = spatial_hints
            logger.info("  → Spatial analysis added to payload: %s of '%s'", 
                       spatial_hints["spatial_direction"], 
                       spatial_hints["reference_label"])
        
        # FIX: Recent actions'ı daha da kısalt
        if recent_actions:
            if self.llm_provider == "ollama":
                last = recent_actions[-1]
                payload["recent_actions"] = [{
                    "action_id": last.get("action_id"),
                    "steps_count": last.get("steps_count"),
                }]
            else:
                payload["recent_actions"] = recent_actions[-2:]  #LM Studio için -2 olbilir?

        if schema_hint:
            payload["backend_guidance"] = schema_hint
        
        if note_to_llm:
            payload["note_to_llm"] = note_to_llm
        
        if attempt_number > 1:
            detection_method = "ODS-enhanced"
            payload["retry_context"] = {
                "attempt": attempt_number,
                "detection_method": detection_method,
                "message": f"Retry attempt {attempt_number} using {detection_method} detection. Previous WinDriver attempt did not find element or achieve expected result.",
            }
        else:
            payload["retry_context"] = {
                "attempt": 1,
                "detection_method": "WinDriver",
                "message": "First attempt using WinDriver detection. If element not found with high confidence, return empty steps for ODS retry.",
            }
        
        user_content = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"<JSON_ONLY>{user_content}</JSON_ONLY>"},
        ]

    # ========================================================================
    # PLAN PARSING
    # ========================================================================

    def _parse_plan(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into action plan"""
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

        # Step count artık run_step() içinde kontrol ediliyor
        # truncate yerine flag set ediyoruz
        if self.max_plan_steps and len(steps) > self.max_plan_steps:
            logger.warning("LLM plan has %d steps; will be rejected in run_step", len(steps))

        if "action_id" not in plan or not plan["action_id"]:
            plan["action_id"] = f"step_{int(time.time() * 1000)}"
        if "coords_space" not in plan:
            plan["coords_space"] = "physical"

        return plan

    def _strip_common_wrappers(self, text: str) -> str:
        """Strip markdown fences and common prefixes"""
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
        """Extract first valid JSON object from text"""
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
        """Slice balanced JSON object from text"""
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
        """Ensure plan has valid steps list"""
        steps = self._normalise_step_candidate(plan.get("steps"))
        if steps is not None:
            return steps, "steps"
        return self._recover_steps_from_plan(plan)

    def _recover_steps_from_plan(self, plan: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Try to recover steps from alternate locations"""
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
        """Normalize steps candidate to list of dicts"""
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

    def _attempt_repair_plan(self, payload: str) -> Optional[str]:
        """Attempt to repair malformed JSON"""
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
        """Close unbalanced brackets and quotes"""
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
        """Remove trailing comma before closing bracket"""
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

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_plan_against_screen(self, plan: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
        """Validate plan coordinates against screen bounds"""
        screen = state.get("screen") or {}
        sw, sh = screen.get("w"), screen.get("h")
        if not isinstance(sw, int) or not isinstance(sh, int) or sw <= 0 or sh <= 0:
            return None

        def in_bounds(x: float, y: float) -> bool:
            return 0 <= x < sw and 0 <= y < sh

        for idx, step in enumerate(plan.get("steps", [])):
            t = step.get("type")
            if t == "click":
                target = step.get("target") or {}
                point = target.get("point")
                if point:
                    px, py = point.get("x"), point.get("y")
                    if any(not isinstance(v, (int, float)) for v in (px, py)):
                        return f"step[{idx}] CLICK point must have numeric x,y"
                    if not in_bounds(px, py):
                        return f"step[{idx}] CLICK point ({px},{py}) outside screen {sw}x{sh}"
                else:
                    return f"step[{idx}] CLICK must have target.point"
                cc = step.get("click_count", 1)
                if not isinstance(cc, int) or cc < 1 or cc > 2:
                    return f"step[{idx}] CLICK click_count must be 1 or 2"
            elif t == "drag":
                p1, p2 = step.get("from") or {}, step.get("to") or {}
                if any(not isinstance(p1.get(k), (int, float)) for k in ("x", "y")) or \
                   any(not isinstance(p2.get(k), (int, float)) for k in ("x", "y")):
                    return f"step[{idx}] DRAG must have numeric from/to"
                if not (in_bounds(p1["x"], p1["y"]) and in_bounds(p2["x"], p2["y"])):
                    return f"step[{idx}] DRAG points outside screen {sw}x{sh}"
            elif t == "scroll":
                at = step.get("at")
                if at:
                    if any(not isinstance(at.get(k), (int, float)) for k in ("x", "y")):
                        return f"step[{idx}] SCROLL.at must have numeric x,y"
                    if not in_bounds(at["x"], at["y"]):
                        return f"step[{idx}] SCROLL.at outside screen {sw}x{sh}"
            elif t in ("type", "key_combo", "wait"):
                pass
            else:
                return f"step[{idx}] unknown action type: {t}"
        return None

    # ========================================================================
    # EXPECTED RESULT VALIDATION
    # ========================================================================

    def _expected_holds(
        self, 
        state: Dict[str, Any], 
        expected_result: str,
        plan_executed: Dict[str, Any],
        state_before: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Enhanced validation: Check BOTH text match AND action completion"""

        if not expected_result:
            return False

        steps_executed = plan_executed.get("steps", [])

        if not steps_executed:
            logger.warning("❌ No actions executed, cannot validate expected result")
            return False

        meaningful_action_types = {"click", "type", "key_combo", "drag"}
        has_meaningful_action = any(
            step.get("type") in meaningful_action_types 
            for step in steps_executed
        )

        if not has_meaningful_action:
            logger.warning("❌ Only passive actions (wait) executed, no meaningful interaction")
            return False

        text = expected_result.lower().strip()
        elems = (state or {}).get("elements", [])

        if not elems:
            return False

        visible_names = []
        for e in elems:
            for key in ["name", "name_ods", "name_ocr"]:
                name = (e.get(key) or "").strip()
                if name:
                    visible_names.append(name.lower())

        if " and " in text:
            conditions = text.split(" and ")
            logger.debug("  → Expected has multiple conditions: %d", len(conditions))

            all_conditions_met = True
            for i, condition in enumerate(conditions, 1):
                condition = condition.strip()
                logger.debug("    Condition %d: %s", i, condition[:50])

                condition_met = self._check_single_condition(
                    condition, 
                    visible_names, 
                    state, 
                    state_before
                )

                if not condition_met:
                    logger.info("❌ Condition %d NOT met: %s", i, condition[:50])
                    all_conditions_met = False
                    break
                
            if all_conditions_met:
                logger.info("✓ Expected HOLDS: All %d conditions met", len(conditions))
                return True
            else:
                return False

        closing_keywords = ["closes", "closed", "disappears", "disappeared", "hidden", "gone"]
        expects_closing = any(kw in text for kw in closing_keywords)

        if expects_closing:
            logger.debug("  → Expected result mentions closing/disappearing")
            closing_targets = self._extract_closing_targets(text)
            targets_absent = self._verify_targets_absent(closing_targets, visible_names)

            if targets_absent:
                logger.info("✓ Expected HOLDS: Target elements successfully closed/removed")
                return True
            else:
                logger.info("❌ Expected closing not achieved: targets still present in UI")
                return False

        visibility_keywords = ["visible", "appears", "shown", "displayed", "open", "opened", "shows"]
        expects_visibility = any(kw in text for kw in visibility_keywords)

        if expects_visibility:
            logger.debug("  → Expected result mentions visibility, checking targets")
            target_tokens = self._extract_visibility_targets(text)
            targets_visible = self._verify_targets_visible(target_tokens, visible_names)

            if not targets_visible:
                logger.info("❌ Expected visibility not achieved: targets not found in UI")
                return False

        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_items = re.findall(quoted_pattern, expected_result)

        if quoted_items:
            all_found = all(
                any(quoted.lower() in name for name in visible_names)
                for quoted in quoted_items
            )

            if all_found:
                selection_keywords = ["selected", "highlighted", "active", "checked"]
                expects_selection = any(kw in expected_result.lower() for kw in selection_keywords)

                if expects_selection:
                    logger.debug("  → Expected mentions selection, verifying UI change")

                    if state_before is None:
                        logger.warning("⚠️ Cannot verify UI change: state_before not provided")
                    else:
                        ui_changed, change_mag = self._detect_ui_change(state_before, state)

                        if not ui_changed:
                            logger.info("❌ Quoted item found but NO UI change for selection")
                            return False
                        else:
                            logger.debug("  ✓ UI changed for selection (%.1f%%)", change_mag * 100)

                process_states = {"running", "started", "active", "stopped", "paused", "executing"}
                expected_lower = expected_result.lower()
                mentioned_process_states = [s for s in process_states if s in expected_lower]

                if mentioned_process_states:
                    state_found = any(s in name for s in mentioned_process_states for name in visible_names)
                    if state_found:
                        logger.info("✓ Expected HOLDS: Quoted items + process state verified")
                        return True
                    else:
                        logger.debug("Quoted found but process state %s missing", mentioned_process_states)
                else:
                    logger.info("✓ Expected HOLDS: All quoted items present")
                    return True
                
        def tokenize_fuzzy(s: str) -> Set[str]:
            tokens = set(re.findall(r'\b\w{3,}\b', s.lower()))
            stopwords = {
                "the", "is", "are", "was", "were", "been", "being",
                "have", "has", "had", "do", "does", "did",
                "will", "would", "should", "could", "may", "might",
                "can", "shall", "and", "or", "but", "not",
                "this", "that", "these", "those", "with", "from",
                "for", "about", "into", "through",
                "observed", "shown", "displayed", "visible", "opened",
                "closed", "clicked", "selected", "entered", "verified",
                "appears", "exists", "present", "contains",
                "it", "that", "which", "who", "where", "when", "how",
            }
            return tokens - stopwords
    
        expected_tokens = tokenize_fuzzy(text)
        visible_tokens = tokenize_fuzzy(" ".join(visible_names))

        if not expected_tokens:
            return any(word in name for word in text.split() if len(word) > 3 for name in visible_names)

        overlap = expected_tokens & visible_tokens
        coverage = len(overlap) / len(expected_tokens) if expected_tokens else 0

        logger.debug("  Keyword coverage: %.1f%% (%d/%d)", coverage * 100, len(overlap), len(expected_tokens))

        threshold = 0.85 if expects_visibility else 0.75

        if coverage >= threshold:
            logger.info("✓ Expected HOLDS: %.1f%% keyword match", coverage * 100)
            return True

        def fuzzy_token_match(expected_tokens: Set[str], visible_tokens: Set[str], threshold: float = 0.80) -> Tuple[bool, float]:
            if not expected_tokens:
                return False, 0.0
            
            matched = 0
            for exp_token in expected_tokens:
                best_ratio = 0.0
                for vis_token in visible_tokens:
                    ratio = SequenceMatcher(None, exp_token, vis_token).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                
                if best_ratio >= threshold:
                    matched += 1
            
            fuzzy_coverage = matched / len(expected_tokens)
            return fuzzy_coverage >= 0.75, fuzzy_coverage
        
        fuzzy_match, fuzzy_coverage = fuzzy_token_match(expected_tokens, visible_tokens, threshold=0.80)
        
        if fuzzy_match:
            logger.info("✓ Expected HOLDS: %.1f%% fuzzy keyword match", fuzzy_coverage * 100)
            return True
        
        def smart_fuzzy_match(
            expected_text: str,
            visible_items: List[str],
            require_action_context: bool = False
        ) -> Tuple[bool, float, str]:
            action_verbs = ["selected", "clicked", "opened", "closed", "changed", "entered", "shows"]
            has_action_verb = any(verb in expected_text.lower() for verb in action_verbs)
            
            threshold = 0.85 if (has_action_verb or require_action_context) else 0.75
            
            if has_action_verb:
                logger.debug("  → Expected mentions action, requiring stricter match (85%%)")
            
            best_match = 0.0
            best_item = ""
            
            expected_entities = self._extract_key_entities(expected_text)
            
            clean_expected = re.sub(r'\b(it is|that|the|observed|shown|displayed)\b', '', expected_text)
            clean_expected = re.sub(r'\s+', ' ', clean_expected).strip()
            
            for item in visible_items:
                item_lower = item.lower()
                
                entities_found = sum(
                    1 for entity in expected_entities 
                    if entity.lower() in item_lower
                )
                
                entity_coverage = entities_found / len(expected_entities) if expected_entities else 0
                
                text_ratio = SequenceMatcher(None, clean_expected.lower(), item_lower).ratio()
                
                combined_score = (text_ratio * 0.6) + (entity_coverage * 0.4)
                
                if combined_score > best_match:
                    best_match = combined_score
                    best_item = item
            
            matched = best_match >= threshold
            return matched, best_match, best_item
        
        fuzzy_matched, fuzzy_score, matched_item = smart_fuzzy_match(text, visible_names, require_action_context=expects_visibility)
        
        if fuzzy_matched:
            logger.info("✓ Expected HOLDS: Fuzzy match '%.30s...' (%.1f%%)", matched_item, fuzzy_score * 100)
            return True
        
        if "running" in text or "started" in text or "active" in text:
            has_stop = any("stop scenario" in name or "pause scenario" in name for name in visible_names)
            has_run = any("run scenario" in name for name in visible_names)
            
            if has_stop and not has_run:
                logger.info("✓ Expected HOLDS: Run button changed to Stop")
                return True
        
        logger.debug("✗ Expected NOT met")
        logger.debug("  Expected tokens: %s", expected_tokens)
        logger.debug("  Visible tokens sample: %s", list(visible_tokens)[:20])
        return False
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    def _extract_closing_targets(self, expected_text: str) -> Set[str]:
        """Extract what elements should close/disappear"""
        closing_keywords = ["closes", "closed", "disappears", "disappeared", "hidden", "gone"]

        for keyword in closing_keywords:
            if keyword in expected_text.lower():
                parts = expected_text.lower().split(keyword)
                if parts:
                    target_text = parts[0].strip()
                    tokens = set(re.findall(r'\b\w{3,}\b', target_text))
                    stopwords = {"the", "is", "are", "was", "and", "or", "will", "should", "must"}
                    return tokens - stopwords

        return set()

    def _verify_targets_absent(
        self, 
        target_tokens: Set[str], 
        visible_names: List[str]
    ) -> bool:
        """Verify that target tokens are NOT present in visible UI"""

        if not target_tokens:
            return True

        found_count = 0

        for token in target_tokens:
            token_found = any(token in visible_name for visible_name in visible_names)

            if token_found:
                found_count += 1
                logger.debug("    → '%s' still present in UI", token)

        presence_ratio = found_count / len(target_tokens)

        logger.debug("  → Closing check: %d/%d targets still present (%.1f%%)", 
                    found_count, len(target_tokens), presence_ratio * 100)

        return presence_ratio < 0.30

    def _extract_visibility_targets(self, expected_text: str) -> Set[str]:
        """Extract what elements should be visible from expected result"""

        # FIX 1: Önce quoted items kontrol et
        quoted_items = re.findall(r'["\']([^"\']+)["\']', expected_text)
        if quoted_items:
            tokens = set()
            for item in quoted_items:
                tokens.update(re.findall(r'\b\w{3,}\b', item.lower()))
            return tokens

        # FIX 2: Multi-word phrases'i koru!
        # "Scenario Management dialog" → ["scenario management", "dialog"]

        # Remove common visibility keywords
        clean = re.sub(
            r'\b(is|are|be|being|appears?|shown|displayed|visible|open|opened|shows)\b',
            '',
            expected_text,
            flags=re.IGNORECASE
        )

        # Remove selection keywords
        clean = re.sub(
            r'\b(selected|highlighted|active|checked|and)\b',
            '',
            clean,
            flags=re.IGNORECASE
        )

        # Remove articles
        clean = re.sub(r'\b(with|the|a|an)\b', '', clean, flags=re.IGNORECASE)

        clean = clean.strip()

        # FIX 3: "dialog", "window", "panel" gibi UI type keywords'ü ayır
        ui_type_keywords = ["dialog", "window", "panel", "form", "popup", "modal"]

        # Extract UI type if present
        ui_type = None
        for keyword in ui_type_keywords:
            if keyword in clean.lower():
                ui_type = keyword
                # Remove UI type from clean text
                clean = re.sub(rf'\b{keyword}\b', '', clean, flags=re.IGNORECASE)
                break
            
        clean = re.sub(r'\s+', ' ', clean).strip()

        # FIX 4: Multi-word entity olarak döndür
        # "scenario management" → tek token olarak tut
        tokens = set()

        if clean:
            # Tüm phrase'i tek token olarak ekle
            tokens.add(clean.lower())

            # Ayrıca individual words'leri de ekle (fallback için)
            individual = re.findall(r'\b\w{3,}\b', clean.lower())
            tokens.update(individual)

        # UI type'ı da ekle (opsiyonel)
        if ui_type:
            tokens.add(ui_type)

        stopwords = {"that", "this", "from", "have", "will", "should", "must"}
        result = tokens - stopwords

        logger.debug("  → Extracted visibility tokens: %s", result)
        return result

    def _verify_targets_visible(
        self, 
        target_tokens: Set[str], 
        visible_names: List[str]
) ->     bool:
        """Verify that target tokens are actually present in visible UI"""

        if not target_tokens:
            return False

        found_count = 0
        phrase_found = False

        # Önce en uzun phrase'leri kontrol et (multi-word matches)
        sorted_tokens = sorted(target_tokens, key=lambda x: -len(x))

        for token in sorted_tokens:
            token_found = False

            #  Exact substring match (case-insensitive)
            for visible_name in visible_names:
                if token in visible_name.lower():
                    token_found = True

                    # Multi-word match
                    if ' ' in token:  
                        phrase_found = True
                        logger.debug("    ✓ EXACT PHRASE MATCH: '%s' in '%s'", token, visible_name)

                    break
                
            # FUZZY match for multi-word phrases (if exact failed)
            if not token_found and ' ' in token:
                # "scenario management dialog" → check if words appear close together
                token_words = token.split()

                for visible_name in visible_names:
                    visible_lower = visible_name.lower()

                    # Check if all words present
                    all_words_present = all(word in visible_lower for word in token_words)

                    if all_words_present:
                        # Check word proximity (optional)
                        token_found = True
                        phrase_found = True
                        logger.debug("    ✓ FUZZY PHRASE: all words of '%s' in '%s'", 
                                   token, visible_name)
                        break            

            if token_found:
                found_count += 1

        coverage = found_count / len(target_tokens) if target_tokens else 0

        logger.debug("  → Visibility: %d/%d found (%.1f%%), phrase_match=%s", 
                found_count, len(target_tokens), coverage * 100, phrase_found)

        # Phrase match varsa threshold düşür
        threshold = 0.50 if phrase_found else 0.70

        success = coverage >= threshold

        if success:
            logger.info("✓ Visibility PASSED: %.1f%% >= %.0f%% (phrase=%s)", 
                       coverage * 100, threshold * 100, phrase_found)
        else:
            logger.info("✗ Visibility FAILED: %.1f%% < %.0f%%", 
                       coverage * 100, threshold * 100)

        return success

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities that should be present"""
        quoted = re.findall(r'["\']([^"\']+)["\']', text)

        entity_patterns = [
            r'\b[A-Z][A-Za-z0-9_-]+\b',
            r'\b\w+\s+\w+\s+(?:panel|tab|dialog|window|button)\b',
        ]

        entities = quoted.copy()
        for pattern in entity_patterns:
            entities.extend(re.findall(pattern, text))

        return list(set(entities))

    def _check_single_condition(
        self,
        condition: str,
        visible_names: List[str],
        state: Dict[str, Any],
        state_before: Optional[Dict[str, Any]],
    ) -> bool:  # FIX: Format düzelt
        """Check a single condition from expected result (for AND logic)"""

        closing_keywords = ["closes", "closed", "disappears", "disappeared", "hidden", "gone"]
        if any(kw in condition for kw in closing_keywords):
            logger.debug("    → Condition expects closing/disappearing")
            closing_targets = self._extract_closing_targets(condition)
            targets_absent = self._verify_targets_absent(closing_targets, visible_names)

            if targets_absent:
                logger.debug("    ✓ Closing condition passed")
                return True
            else:
                if state_before is not None:
                    ui_changed, change_mag = self._detect_ui_change(state_before, state)
                    if ui_changed and change_mag > 0.2:
                        logger.debug("    ✓ Closing: UI changed significantly (%.1f%%), assuming closed", change_mag * 100)
                        return True

                logger.debug("    ✗ Closing condition failed (targets still present)")
                return False

        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_items = re.findall(quoted_pattern, condition)
        quoted_items = [item.replace('\\', '').strip() for item in quoted_items]

        logger.info("    → Condition: '%s'", condition[:80])
        logger.info("    → Quoted items found: %s", quoted_items)

        if quoted_items:
            all_found = all(
                any(quoted.lower() in name for name in visible_names)
                for quoted in quoted_items
            )

            logger.info("    → All quoted items found in UI: %s", all_found)

            if not all_found:
                logger.debug("    → Quoted items not found in condition")
                return False

            selection_keywords = ["selected", "highlighted", "active", "checked"]
            has_selection_keyword = any(kw in condition.lower() for kw in selection_keywords)

            logger.info("    → Has selection keyword: %s", has_selection_keyword)

            if has_selection_keyword:
                logger.info("    → Selection keyword detected in condition")

                if state_before is not None:
                    ui_changed, change_mag = self._detect_ui_change(state_before, state)

                    logger.info("    → UI change check: %s (%.1f%%)", 
                               "YES" if ui_changed else "NO", 
                               change_mag * 100)

                    if ui_changed and change_mag > 0.20:
                        logger.info("    ✓ Selection condition passed (UI changed + quoted found)")
                        return True
                    else:
                        logger.info("    ✗ Selection expected but insufficient UI change (%.1f%% < 20%%)", change_mag * 100)
                        return False
                else:
                    logger.info("     No state_before, assuming selection based on quoted item")
                    return True

            logger.info("    ✓ Quoted items condition passed")
            return True

        visibility_keywords = ["visible", "appears", "shown", "displayed", "open", "opened", "panel"]
        if any(kw in condition for kw in visibility_keywords):
            target_tokens = self._extract_visibility_targets_simple(condition)

            if not target_tokens:
                logger.debug("    → No target tokens extracted from condition")
                return False

            found_count = sum(
                1 for token in target_tokens
                if any(token in name for name in visible_names)
            )

            coverage = found_count / len(target_tokens) if target_tokens else 0
            logger.info("    → Visibility: %d/%d tokens found (%.1f%%)", 
                        found_count, len(target_tokens), coverage * 100)

            if coverage >= 0.5:
                logger.info("    ✓ Visibility condition passed")
                return True
            else:
                logger.info("    ✗ Visibility condition failed")
                return False

        return self._fuzzy_match_condition(condition, visible_names)

    def _extract_visibility_targets_simple(self, condition: str) -> Set[str]:
        """Extract target tokens from visibility condition"""

        clean = re.sub(
            r'\b(is|are|be|being|appears?|shown|displayed|visible|open|opened|shows|the|a|an|with|and)\b',
            '',
            condition,
            flags=re.IGNORECASE
        )

        tokens = set(re.findall(r'\b\w{3,}\b', clean.lower()))

        stopwords = {"that", "this", "from", "have", "will", "should", "must", "entity", "item"}

        result = tokens - stopwords

        logger.debug("    → Extracted visibility tokens: %s", result)
        return result

    def _fuzzy_match_condition(self, condition: str, visible_names: List[str]) -> bool:
        """Fuzzy match for condition"""
        tokens = set(re.findall(r'\b\w{3,}\b', condition.lower()))
        stopwords = {"is", "are", "the", "and", "or", "with", "that", "this"}
        tokens = tokens - stopwords

        if not tokens:
            logger.debug("    → No tokens for fuzzy match")
            return False

        found = sum(
            1 for token in tokens
            if any(token in name for name in visible_names)
        )

        coverage = found / len(tokens) if tokens else 0
        logger.debug("    → Fuzzy match: %d/%d tokens (%.1f%%)", found, len(tokens), coverage * 100)

        return coverage >= 0.4

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _summarise_for_prompt(self, entry: ActionExecutionLog) -> Dict[str, Any]:
        """Summarize action log for prompt"""
        summary: Dict[str, Any] = {
            "action_id": entry.action_id,
            "steps_count": len(entry.plan.get("steps", [])),
            "ack_status": entry.ack.get("status"),
            "timestamp": entry.timestamp,
        }

        if entry.ack.get("status") != "ok":
            summary["ack_error"] = entry.ack.get("message", "")

        if entry.state_after is not None:
            summary["result_element_count"] = len(entry.state_after.get("elements", []))
        return summary

    def _prune_state_blobs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Remove large blobs from state"""
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


# ============================================================================
# EXPORTS
# ============================================================================

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
    "SYSTEM_PROMPT",
]