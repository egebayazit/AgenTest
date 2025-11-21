# llm_backend.py
# Core LLM orchestration for AgenTest - WITH OLLAMA SUPPORT

from __future__ import annotations

import asyncio
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
# PRODUCTION SYSTEM PROMPT - UPDATED WITH COMPREHENSIVE SPATIAL SUPPORT
# ============================================================================

PRODUCTION_SYSTEM_PROMPT = """You are AgenTest, an expert UI automation planner for Windows applications.

# YOUR MISSION
Convert manual test steps into precise action plans using COORDINATES as your PRIMARY guide.
CRITICAL JSON FORMAT REQUIREMENT 
You MUST respond with ONLY a valid JSON object. NO other text before or after.

REQUIRED structure:
{
  "action_id": "step_1234567890",
  "coords_space": "physical",
  "steps": [
    {"type": "click", "button": "left", "click_count": 1, "target": {"point": {"x": 278, "y": 619}}},
    {"type": "wait", "ms": 800}
  ],
  "reasoning": "Found communication_test at (278, 619)"
}

⚠️ CRITICAL: "steps" field MUST be an array (even if empty: [])
⚠️ NEVER return {"steps": null} or omit "steps" field
⚠️ If element not found, return: {"steps": [], "reasoning": "Element not found"}

# ⚠️ CRITICAL: COORDINATES ARE KING
- UI element names (name, name_ocr, name_ods) may be INCOMPLETE, PARTIAL, or WRONG
- ALWAYS prioritize SPATIAL CONTEXT (coordinates, nearby elements) over exact text matching
- Example: If test_step says "Click Force ID dropdown" but you only see "Force" at coordinates,
  and there's a dropdown-like element nearby → USE THAT COORDINATE

# CRITICAL RULE: DO NOT ACT IF ELEMENT NOT FOUND
If you cannot find the target element with REASONABLE CONFIDENCE:
- Return EMPTY steps: {"steps": []}
- The system will automatically retry with enhanced detection (ODS)

NEVER guess coordinates or click generic elements hoping they're correct!

# INPUT FORMAT
You receive:
- test_step: The CURRENT manual instruction (FOCUS ONLY ON THIS)
- expected_result: Success criteria for THIS STEP ONLY
- current_state: Available UI elements with their exact click coordinates
- spatial_analysis: Nearby element analysis with directional information (when available)
- recent_actions: Previously executed actions (DO NOT repeat unless they failed)
- screen: Display bounds for coordinate validation
- retry_context: Which attempt (1=WinDriver, 2=ODS enhanced)

# ELEMENT STRUCTURE
Each element has:
{
  "name": "Button text",              // Original UI text (MOST RELIABLE but RARE)
  "name_ocr": "Detected text",        // OCR result (MAY BE INCOMPLETE)
  "name_ods": "Parsed text",          // OmniParser result (MAY BE PARTIAL)
  "center": {"x": 1561, "y": 797}     //  EXACT CLICK COORDINATES (MOST RELIABLE)
}

# ⚠️ NAME FIELD QUALITY WARNING
- **name**: Perfect if available (RARE - Windows native controls only)
- **name_ods**: Often INCOMPLETE or PARTIAL (e.g., "Force ID" may appear as "Force")
- **name_ocr**: May have OCR errors or be PARTIAL (e.g., "Button" might be "Buttor")
- **CRITICAL**: Use names as HINTS only, but TRUST COORDINATES + SPATIAL CONTEXT

# CRITICAL ELEMENT SELECTION ALGORITHM

## Step 1: Extract Target Keywords from CURRENT test_step ONLY
Example: "Click dropdown next to Force ID"
→ Keywords: ["dropdown", "Force", "ID"]
→ Reference: "Force ID" (or partial like "Force")
→ Spatial: "next to" (RIGHT direction)

⚠️ IGNORE any other steps or future actions. Focus ONLY on current test_step!

## Step 2: Check for spatial_analysis (PRIMARY METHOD)
If spatial_analysis is provided in input, USE IT as your PRIMARY guide:
- reference_label: The label mentioned in test_step (may be partial match)
- reference_location: Coordinates of the label
- spatial_direction: Direction to search (right, left, above, below, near, same_row, same_column)
- nearby_candidates: Elements in that direction (sorted by distance)
- hint: Human-readable description

**Supported Directions:**
- **RIGHT**: "next to", "beside", "right of", "to the right of", "adjacent to"
- **LEFT**: "left of", "to the left of", "on the left of"
- **ABOVE**: "above", "over", "top of", "upper"
- **BELOW**: "below", "under", "beneath", "bottom of", "lower"
- **NEAR**: "near", "close to", "around", "nearby"
- **SAME_ROW**: "same row as"
- **SAME_COLUMN**: "same column as"

**Algorithm when spatial_analysis exists:**
1. Review spatial_direction field
2. Review nearby_candidates list (pre-filtered, sorted by distance)
3. Select FIRST candidate that is:
   - NOT a label (no ":" at end)
   - NOT a column header
   - Has value-like text or is a control element
4. Use that candidate's coordinates

## Step 3: Fallback to Fuzzy Text Matching (if no spatial_analysis)

### ⚠️ ACCEPT PARTIAL MATCHES
- "Force" CAN match "Force ID"
- "Active" CAN match "Active status"
- "ID" CAN match "Force ID dropdown"
- Case insensitive matching
- ANY keyword overlap is acceptable

### Search Priority (IN THIS ORDER):
1. **Partial match on "name" field** (if available)
   - ✅ "Force" matches element with name="Force ID"
   
2. **Partial match on "name_ods" field** (OmniParser)
   - ✅ "Force" matches element with name_ods="Force"
   - ⚠️ name_ods is often incomplete, so be tolerant
   
3. **Partial match on "name_ocr" field** (OCR)
   - ✅ "ID" matches element with name_ocr="Force ID"
   
4. **Keyword overlap**
   - Split target into words
   - Match if ANY word appears in element name
   - ✅ "dropdown" in test_step matches name_ods="Force" if nearby

5. **Coordinate-based reasoning**
   - If multiple partial matches, choose closest to expected position
   - Prefer elements with larger clickable area
   - Validate coordinates are within screen bounds

### Priority 1b: CRITICAL - Spatial phrase detected but NO spatial_analysis

⚠️ **If test_step contains spatial phrases (next to, beside, etc.) BUT spatial_analysis is NOT provided:**

This means the reference label was NOT FOUND in current state.

→ Return EMPTY STEPS: {"steps": []}
→ Reasoning: "Spatial relationship detected but reference not found. Waiting for ODS retry."

## Step 4: REJECT These Elements (Even if text matches)

❌ REJECT: Generic UI labels
- "View", "Text", "Line", "ImageView" (container types)
- "Name", "Description", "Status" (column headers)

❌ REJECT: Internal identifiers
- "uiDataValue", "qMenuBarMenu" (internal field names)
- "Default IME", "MSCTFIME UI" (system components)

❌ REJECT: Off-screen elements
- Elements at (0, 0) unless confirmed valid
- Elements outside screen bounds

❌ REJECT: Labels when looking for interactive elements
- Text ending with ":"
- Column headers in tables

## Step 5: ELEMENT NOT FOUND → RETURN EMPTY STEPS
❌ If still no match found, DO NOT use keyboard shortcuts as fallback
❌ DO NOT guess or estimate coordinates
✅ Return {"steps": [], "reasoning": "Target element 'X' not found in current state"}

# OUTPUT SCHEMA
{
  "action_id": "step_<timestamp>",
  "coords_space": "physical",
  "steps": [
    // EMPTY ARRAY if element not found with reasonable confidence
    // OR click at exact point if found
    {"type": "click", "button": "left", "click_count": 1, 
     "target": {"point": {"x": 1561, "y": 797}}},
    
    // WAIT after UI changes
    {"type": "wait", "ms": 800},
    
    // TYPE text (only if element found and focused)
    {"type": "type", "text": "search query", "delay_ms": 30, "enter": false},
    
    // KEY_COMBO (only for confirmed shortcuts, not fallbacks)
    {"type": "key_combo", "combo": ["ctrl", "o"]},
    
    // Other actions: drag, scroll, move (rarely needed)
  ],
  "reasoning": "Brief explanation: HOW you found the element (spatial/partial/coordinate match) - max 100 words"
}

# CRITICAL RULES - READ CAREFULLY
1. Output MUST be valid JSON (no markdown fences)
2. ⚠️ **COORDINATES > TEXT MATCHING** - Always prioritize spatial context
3. ⚠️ **ACCEPT PARTIAL/INCOMPLETE NAMES** - "Force" can match "Force ID"
4. When spatial_analysis is provided, ALWAYS use it as PRIMARY guide
5. Use spatial_direction to understand the relationship
6. Select FIRST candidate from nearby_candidates unless specific reason not to
7. Use ONLY coordinates from current_state elements OR spatial_analysis.nearby_candidates
8. ALL coordinates must be within screen bounds
9. Max 24 steps per plan
10. ⚠️ CRITICAL: If element NOT FOUND with reasonable confidence → Return EMPTY steps
11. ⚠️ CRITICAL: Focus ONLY on current test_step, ignore other steps
12. DO NOT repeat actions from recent_actions (unless they failed)
13. NEVER use keyboard shortcuts as fallback when element not found

# EXAMPLES OF ACCEPTABLE PARTIAL MATCHES

**Example 1: Partial name match**
- test_step: "Click Force ID dropdown"
- current_state has: {"name_ods": "Force", "center": {"x": 100, "y": 200}}
- ✅ ACCEPT: Click at (100, 200) - "Force" is partial match for "Force ID"
- reasoning: "Found partial match: 'Force' matches target 'Force ID' at coordinates (100, 200)"

**Example 2: Spatial match with partial name**
- test_step: "Click dropdown next to Force ID"
- spatial_analysis: {"reference_label": "Force", "nearby_candidates": [{"name_ods": "", "center": {"x": 150, "y": 200}}]}
- ✅ ACCEPT: Click at (150, 200) - spatial context is PRIMARY
- reasoning: "Found via spatial analysis: element RIGHT of 'Force' label"

**Example 3: Keyword overlap**
- test_step: "Select Active status"
- current_state has: {"name_ocr": "Active", "center": {"x": 300, "y": 400}}
- ✅ ACCEPT: Click at (300, 400) - "Active" is key word match
- reasoning: "Found keyword match: 'Active' at coordinates (300, 400)"

**Example 4: Empty name but correct position**
- test_step: "Click button next to Name label"
- spatial_analysis finds element RIGHT of "Name" with name_ods=""
- ✅ ACCEPT: Use coordinates from nearby_candidates[0]
- reasoning: "Found via spatial analysis: interactive element RIGHT of 'Name' label (name may be empty but position is correct)"

# OUTPUT FORMAT
Return ONLY the JSON object, no other text:
{
  "action_id": "step_<timestamp>",
  "coords_space": "physical",
  "steps": [...],  // EMPTY if element not found
  "reasoning": "Found via [spatial/partial/coordinate] match: ..."
}
"""

DEFAULT_SYSTEM_PROMPT = PRODUCTION_SYSTEM_PROMPT

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
            "maxItems": 24,
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
                    {  # MOVE
                        "properties": {
                            "type": {"const": "move"},
                            "point": {
                                "type": "object",
                                "required": ["x", "y"],
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                            },
                            "settle_ms": {"type": "integer", "minimum": 0},
                        },
                        "required": ["type", "point"],
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
# LLM BACKEND - WITH OLLAMA SUPPORT
# ============================================================================

class LLMBackend:
    """LLM-based action planner backend with Ollama and OpenRouter support"""
    
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
        llm_model: str ,
        llm_base_url: str = "http://localhost:11434",
        llm_api_key: Optional[str] = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_attempts: int = 2,
        post_action_delay: float = 0.5,
        sut_timeout: float = 50.0,
        llm_timeout: float = 150.0,
        max_tokens: int = 600,
        max_plan_steps: int = 24,
        schema_retry_limit: int = 1,
        http_referrer: str = "https://agentest.local/backend",
        client_title: str = "AgenTest LLM Backend",
        omit_large_blobs: bool = True,
        enforce_json_response: bool = True,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validation
        if not llm_model:
            raise ValueError("llm_model is required")
        
        if llm_provider not in ("ollama", "openrouter"):
            raise ValueError("llm_provider must be 'ollama' or 'openrouter'")
        
        if llm_provider == "openrouter" and not llm_api_key:
            raise ValueError("llm_api_key required for OpenRouter")
        
        if max_plan_steps <= 0:
            raise ValueError("max_plan_steps must be positive")
        if schema_retry_limit < 0:
            raise ValueError("schema_retry_limit must be >= 0")

        # Store config
        self.state_url_windriver = state_url_windriver
        self.state_url_ods = state_url_ods
        self.max_attempts = 2
        
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
        self.json_schema = json_schema
        
        self._last_element_count = 0

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
            # State endpoints
            "state_url_windriver": overrides.pop(
                "state_url_windriver", 
                env("SUT_STATE_URL_WINDRIVER", "http://127.0.0.1:18800/state/for-llm")
            ),
            "state_url_ods": overrides.pop(
                "state_url_ods", 
                env("SUT_STATE_URL_ODS", "http://127.0.0.1:18800/state/from-ods")
            ),
            
            # Action endpoint
            "action_url": overrides.pop(
                "action_url", 
                env("SUT_ACTION_URL", "http://192.168.137.249:18080/action")
            ),
            
            # LLM config
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
            
            # Other config
            "enforce_json_response": _env_bool("ENFORCE_JSON_RESPONSE"),
        }
        
        params.update(overrides)
        return cls(**params)

    # ========================================================================
    # SPATIAL ANALYSIS IMPLEMENTATION
    # ========================================================================

    def _generate_spatial_hints(
        self, 
        test_step: str, 
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate spatial relationship hints"""
        # 1. DETECT SPATIAL RELATIONSHIP TYPE
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
        
        # 2. FIND REFERENCE ELEMENT
        elements = state.get("elements", [])
        ref_elem = None
        
        # Try exact match first (case-insensitive)
        for elem in elements:
            name = elem.get("name", "").strip()
            if name.lower() == reference_label.lower():
                ref_elem = elem
                logger.debug("  → Exact match: '%s'", name)
                break
        
        # Try partial match if exact not found
        if not ref_elem:
            for elem in elements:
                name = elem.get("name", "").strip()
                if reference_label.lower() in name.lower():
                    ref_elem = elem
                    logger.debug("  → Partial match: '%s'", name)
                    break
        
        # Try fuzzy match if still not found (70% similarity threshold)
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
        
        # 3. SEARCH BASED ON DIRECTION
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
        
        # 4. SORT BY DISTANCE
        nearby_elements.sort(key=lambda e: e["distance"])
        
        logger.debug(
            "  → Spatial hint: Found %d elements %s of '%s'",
            len(nearby_elements),
            detected_direction,
            reference_label
        )
        
        # 5. BUILD RESULT
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
            "nearby_candidates": nearby_elements[:10],
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
            
            # Apply direction-specific filters
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
        """Run a single test step with 2-attempt strategy"""
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

            logger.info("  Plan: %d steps, reasoning: %s", len(steps), reasoning[:100])

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
            
            if self._expected_holds(final_state, expected_result):
                logger.info("✓ Success: Expected result holds after attempt %d (%s)", attempt, detection_method)
                return RunResult(
                    status="passed",
                    attempts=attempt,
                    actions=actions_log,
                    final_state=final_state,
                    last_plan=plan,
                    reason=f"Expected result achieved (detection: {detection_method})",
                )
            else:
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
                
                logger.info("  Retrying with ODS enhanced detection...")

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
    # LLM COMMUNICATION - WITH OLLAMA SUPPORT
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

        # OLLAMA vs OPENROUTER routing
        if self.llm_provider == "ollama":
            return await self._request_plan_ollama(messages, temperature)
        else:
            return await self._request_plan_openrouter(messages, temperature)

    async def _request_plan_ollama(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
) ->     Dict[str, Any]:
        """Request plan from Ollama (local) with JSON Schema enforcement"""
        
        # ============ JSON SCHEMA ENFORCEMENT ============
        json_schema = {
            "type": "object",
            "required": ["action_id", "coords_space", "steps", "reasoning"],
            "properties": {
                "action_id": {"type": "string"},
                "coords_space": {"type": "string", "enum": ["physical"]},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string", "enum": ["click", "wait", "type", "key_combo"]},
                            "button": {"type": "string", "enum": ["left", "right", "middle"]},
                            "click_count": {"type": "integer", "minimum": 1, "maximum": 2},
                            "target": {
                                "type": "object",
                                "required": ["point"],
                                "properties": {
                                    "point": {
                                        "type": "object",
                                        "required": ["x", "y"],
                                        "properties": {
                                            "x": {"type": "number"},
                                            "y": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "reasoning": {"type": "string"}
            }
        }
        logger.info("📋 Using JSON schema enforcement for Ollama")
        # =================================================
        
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": json_schema,  # ← JSON SCHEMA!
            "options": {
                "temperature": temperature,
                "num_predict": self.max_tokens,
            },
        }
        
        url = f"{self.llm_base_url}/api/chat"
        
        logger.error("🚀 SENDING TO OLLAMA:")
        logger.error("URL: %s", url)
        logger.error("Body keys: %s", list(body.keys()))
        logger.error("Format field: %s", str(body.get("format", "MISSING"))[:200])
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
        
        logger.error("🔍 RAW LLM RESPONSE (first 500 chars):")
        logger.error(content[:500])
        logger.error("=" * 80)
    
        if not content:
            raise PlanParseError("Ollama returned empty content")
        
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
            if self.json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "AgenTestPlan", "schema": self.json_schema},
                }
            else:
                body["response_format"] = {"type": "json_object"}
        
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
        
        if recent_actions:
        # Ollama için recent_actions'ı daha da kısalt (JSON bozulmasını önle)
            if self.llm_provider == "ollama":
                payload["recent_actions"] = recent_actions[-1:]  # Sadece son 1
            else:
                payload["recent_actions"] = recent_actions[-3:] 

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
    # PLAN PARSING (remains the same as original)
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

        if self.max_plan_steps and len(steps) > self.max_plan_steps:
            logger.warning("LLM plan has %d steps; truncating to %d", len(steps), self.max_plan_steps)
            plan["steps"] = steps[:self.max_plan_steps]
            reasoning = plan.get("reasoning")
            note = f"Truncated to first {self.max_plan_steps} steps by backend."
            plan["reasoning"] = f"{reasoning} | {note}" if reasoning else note

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
            elif t == "move":
                pt = step.get("point") or {}
                if any(not isinstance(pt.get(k), (int, float)) for k in ("x", "y")):
                    return f"step[{idx}] MOVE.point must have numeric x,y"
                if not in_bounds(pt["x"], pt["y"]):
                    return f"step[{idx}] MOVE.point outside screen {sw}x{sh}"
            elif t in ("type", "key_combo", "wait"):
                pass
            else:
                return f"step[{idx}] unknown action type: {t}"
        return None

    # ========================================================================
    # EXPECTED RESULT CHECK
    # ========================================================================

    def _expected_holds(self, state: Dict[str, Any], expected_result: str) -> bool:
        """Enhanced expected result checking with fuzzy tolerance"""
        if not expected_result:
            return False
        
        text = expected_result.lower().strip()
        elems = (state or {}).get("elements", [])
        
        if not elems:
            return False
        
        # ✅ Tüm name kaynaklarını topla (name, name_ods, name_ocr)
        visible_names = []
        for e in elems:
            for key in ["name", "name_ods", "name_ocr"]:
                name = (e.get(key) or "").strip()
                if name:
                    visible_names.append(name.lower())
        
        # Dialog detection
        has_cancel = any("cancel" in name for name in visible_names)
        has_action_button = any(name in ["ok", "run", "yes", "apply"] for name in visible_names)
        dialog_detected = (has_cancel and has_action_button)
        
        if dialog_detected:
            logger.debug("⚠️ Confirmation dialog detected")
            process_states = {"running", "started", "active", "executing", "processing"}
            mentions_process = any(state in text for state in process_states)
            if mentions_process:
                logger.info("✗ Dialog present but expected mentions process state → NOT met")
                return False
        
        # Quoted items check
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_items = re.findall(quoted_pattern, expected_result)
        
        if quoted_items:
            all_found = all(
                any(quoted.lower() in name for name in visible_names)
                for quoted in quoted_items
            )
            
            if all_found:
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
        
        # ✅ FUZZY TOKENIZATION - Typo toleranslı
        def tokenize_fuzzy(s: str) -> Set[str]:
            """Tokenize with typo tolerance"""
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
        
        # ✅ EXACT MATCH
        overlap = expected_tokens & visible_tokens
        coverage = len(overlap) / len(expected_tokens) if expected_tokens else 0
        
        logger.debug("Keyword coverage: %.1f%% (%d/%d)", coverage * 100, len(overlap), len(expected_tokens))
        
        if coverage >= 0.60:  # ← 0.70'ten 0.60'a düşürdüm
            logger.info("✓ Expected HOLDS: %.1f%% keyword match", coverage * 100)
            return True
        
        # ✅ FUZZY MATCH - Typo toleranslı
        def fuzzy_token_match(expected_tokens: Set[str], visible_tokens: Set[str], threshold: float = 0.75) -> Tuple[bool, float]:
            """Match tokens with typo tolerance"""
            if not expected_tokens:
                return False, 0.0
            
            matched = 0
            for exp_token in expected_tokens:
                best_ratio = 0.0
                for vis_token in visible_tokens:
                    # Levenshtein distance approximation
                    ratio = SequenceMatcher(None, exp_token, vis_token).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                
                # Accept if > 75% similar (typo tolerance)
                if best_ratio >= threshold:
                    matched += 1
            
            fuzzy_coverage = matched / len(expected_tokens)
            return fuzzy_coverage >= 0.60, fuzzy_coverage
        
        fuzzy_match, fuzzy_coverage = fuzzy_token_match(expected_tokens, visible_tokens, threshold=0.75)
        
        if fuzzy_match:
            logger.info("✓ Expected HOLDS: %.1f%% fuzzy keyword match (typo-tolerant)", fuzzy_coverage * 100)
            return True
        
        # ✅ PARTIAL STRING MATCH
        def smart_fuzzy_match(
            expected_text: str,
            visible_items: List[str],
            threshold: float = 0.70  # ← 0.80'den 0.70'e düşürdüm
        ) -> Tuple[bool, float, str]:
            best_match = 0.0
            best_item = ""
            
            process_states = {
                "running", "stopped", "paused", "started", "finished",
                "active", "inactive", "executing", "completed", "ready"
            }
            
            expected_lower = expected_text.lower()
            mentioned_process = [s for s in process_states if s in expected_lower]
            
            clean_expected = re.sub(r'\b(it is|that|the|observed|shown|displayed)\b', '', expected_text)
            clean_expected = re.sub(r'\s+', ' ', clean_expected).strip()
            
            for item in visible_items:
                item_lower = item.lower()
                ratio = SequenceMatcher(None, clean_expected.lower(), item_lower).ratio()
                
                if clean_expected.lower() in item_lower or item_lower in clean_expected.lower():
                    ratio = max(ratio, 0.70)  # ← 0.75'ten 0.70'e
                
                if mentioned_process:
                    has_process_state = any(s in item_lower for s in mentioned_process)
                    if not has_process_state:
                        ratio *= 0.6
                
                if ratio > best_match:
                    best_match = ratio
                    best_item = item
            
            matched = best_match >= threshold
            return matched, best_match, best_item
        
        fuzzy_matched, fuzzy_score, matched_item = smart_fuzzy_match(text, visible_names, threshold=0.70)
        
        if fuzzy_matched:
            logger.info("✓ Expected HOLDS: Fuzzy match '%.30s...' (%.1f%%)", matched_item, fuzzy_score * 100)
            return True
        
        # Button state change detection
        if "running" in text or "started" in text or "active" in text:
            has_stop = any("stop scenario" in name or "pause scenario" in name for name in visible_names)
            has_run = any("run scenario" in name for name in visible_names)
            
            if has_stop and not has_run:
                logger.info("✓ Expected HOLDS: Run button changed to Stop")
                return True
            
            current_count = len(elems)
            previous_count = self._last_element_count
            
            if previous_count > 0:
                change_ratio = (current_count - previous_count) / max(previous_count, 1)
                if change_ratio > 0.10:
                    logger.info("✓ Expected HOLDS: Element count increased %.1f%%", change_ratio * 100)
                    self._last_element_count = current_count
                    return True
        
        self._last_element_count = len(elems)
        
        logger.debug("✗ Expected NOT met")
        logger.debug("  Expected tokens: %s", expected_tokens)
        logger.debug("  Visible tokens sample: %s", list(visible_tokens)[:20])
        return False

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
    "PRODUCTION_SYSTEM_PROMPT",
]