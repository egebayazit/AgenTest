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

from semantic_filter import SemanticStateFilter

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
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are a Windows UI automation assistant. Execute test steps using ONLY elements from current_state.
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
4. Read note_to_llm for element name hints

STATE: (x,y) [Type] Name | Value | AutomationID

ACTIONS:
click: {"type":"click","button":"left","click_count":1,"target":{"point":{"x":INT,"y":INT}}}
type: {"type":"type","text":"STR","enter":true|false}
key_combo: {"type":"key_combo","combo":["ctrl","c"]}
drag: {"type":"drag","from":{"x":INT,"y":INT},"to":{"x":INT,"y":INT}}
scroll: {"type":"scroll","delta":INT,"at":{"x":INT,"y":INT}} (+up,-down)
wait: {"type":"wait","ms":INT}

OUTPUT: {"reasoning":"...","action_id":"step_N","coords_space":"physical","steps":[...]}

EXAMPLES:
Q: Click right of Settings | State: Settings(100,50), Save(150,50)
{"reasoning":"Right of Settings(100). Save(150) nearest.","action_id":"step_1","coords_space":"physical","steps":[{"type":"click","button":"left","click_count":1,"target":{"point":{"x":150,"y":50}}}]}

Q: Click top-right element | State: Menu(1800,40), Logo(50,30)
{"reasoning":"Menu has largest X(1800) and smallest Y(40)=top-right.","action_id":"step_2","coords_space":"physical","steps":[{"type":"click","button":"left","click_count":1,"target":{"point":{"x":1800,"y":40}}}]}

Q: Type hello, Enter
{"reasoning":"Type+Enter.","action_id":"step_3","coords_space":"physical","steps":[{"type":"type","text":"hello","enter":true}]}

Q: Ctrl+S
{"reasoning":"Shortcut.","action_id":"step_4","coords_space":"physical","steps":[{"type":"key_combo","combo":["ctrl","s"]}]}

Q: Click Input then type test | State: Input(200,100)
{"reasoning":"Click first, then type.","action_id":"step_5","coords_space":"physical","steps":[{"type":"click","button":"left","click_count":1,"target":{"point":{"x":200,"y":100}}},{"type":"type","text":"test","enter":false}]}

Q: Double-click File | State: File(300,200)
{"reasoning":"Double-click to open.","action_id":"step_6","coords_space":"physical","steps":[{"type":"click","button":"left","click_count":2,"target":{"point":{"x":300,"y":200}}}]}

Q: Right-click Item | State: Item(400,300)
{"reasoning":"Right-click for context menu.","action_id":"step_7","coords_space":"physical","steps":[{"type":"click","button":"right","click_count":1,"target":{"point":{"x":400,"y":300}}}]}

Q: Drag from A to B | State: A(100,100), B(500,500)
{"reasoning":"Drag A to B.","action_id":"step_8","coords_space":"physical","steps":[{"type":"drag","from":{"x":100,"y":100},"to":{"x":500,"y":500}}]}

Q: Scroll down | State: List(400,400)
{"reasoning":"Negative delta=scroll down.","action_id":"step_9","coords_space":"physical","steps":[{"type":"scroll","delta":-120,"at":{"x":400,"y":400}}]}
"""

# ============================================================================
# LOCAL SYSTEM PROMPT (for Ollama Qwen - Default Local Provider)
# ============================================================================

LOCAL_SYSTEM_PROMPT = """You are a Windows UI automation expert. Execute test steps using ONLY elements from current_state.

OUTPUT: Valid JSON only. No markdown, no explanations.

CURRENT_STATE FORMAT:
ID | Name | Type | (x,y)
------------------------------------------------------------
1 | Save Settings | text | (10,20)
2 | Trash Bin | icon | (100,200)

CORE PRINCIPLE - SPATIAL IS KING:
If user says "next to", "near", "right of", "left of", "below", "above" -> PROXIMITY is PRIMARY.
- "right of A": Find elements where X > Ax AND Y similar to Ay (within 50px). Pick nearest.
- "left of A": Find elements where X < Ax AND Y similar to Ay (within 50px). Pick nearest.
- "below A": Find elements where Y > Ay AND X similar to Ax (within 50px). Pick nearest.
- "above A": Find elements where Y < Ay AND X similar to Ax (within 50px). Pick nearest.
- CHAINED (3 elements): "Click T right of A" + note_to_llm "A is below B" means:
  1. Find B (reference), 2. Find A below B (anchor), 3. Find T right of A (target). Click T's coordinates!
If NO spatial words -> Use exact NAME match.

DUPLICATE NAMES:
If multiple elements have same name (e.g., two 'Input'):
- Use note_to_llm spatial hints to pick the correct one
- "below X" = the one with Y > Xy
- "above X" = the one with Y < Xy

PRIORITY: Check note_to_llm FIRST!
Read note_to_llm for hints about which element to select. Apply spatial rules using those hints.

ACTION RULES:
1. 'click': Requires "target": {"point": {"x":..., "y":...}}.
   - Optional: "button": "right" for context menus.
   - Optional: "click_count": 2 for double-click.
2. 'type': Required: "text": "string". Optional: "enter": true.
3. 'key_combo': Format: "combo": ["ctrl", "c"] or ["enter"].
4. 'drag': Required: "from": {"x":..., "y":...}, "to": {"x":..., "y":...}.
5. 'scroll': Required: "delta": integer (+Up, -Down). Optional: "at": {"x":..., "y":...}.
6. 'wait': Required: "ms": integer.

CRITICAL COORDINATE RULE - VERIFY BEFORE RESPONDING:
1. Find the element by NAME in current_state
2. COPY its EXACT (x,y) coordinates from the state

NEVER invent coordinates. NEVER use coordinates from a different element.
SINGLE TARGET: If instruction says "click X", find X's row, use X's coordinates. Do not use coordinates from other rows.

EXAMPLES:

EXAMPLE 1 — Click right of anchor:
Input: Click right of 'Settings' | State: Settings(100,50), Save(150,50), Cancel(200,100)
{
  "action_id": "step_1",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":1,"target":{"point":{"x":150,"y":50}}}],
  "reasoning": "Anchor Settings at (100,50). Right=X>100, same Y. Save(150,50) nearest."
}

EXAMPLE 2 — Click with note_to_llm hint:
Input: Click right of 'Radar Cross Section Table' | State: Table(997,379), MaxWindow(1241,379), Value(1240,341) | note_to_llm: Target may be formViewField or Maximize window
{
  "action_id": "step_2",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":1,"target":{"point":{"x":1241,"y":379}}}],
  "reasoning": "note_to_llm says Maximize window. Anchor at (997,379). MaxWindow(1241,379) has same Y, X>997."
}

EXAMPLE 3 — Type with Enter:
{
  "action_id": "step_3",
  "coords_space": "physical",
  "steps": [{"type":"type","text":"Hello","enter":true}],
  "reasoning": "Typed text and submitted."
}

EXAMPLE 4 — Click then type:
Input: Click Input then type test | State: Input(200,100)
{
  "action_id": "step_4",
  "coords_space": "physical",
  "steps": [
    {"type":"click","button":"left","click_count":1,"target":{"point":{"x":200,"y":100}}},
    {"type":"type","text":"test","enter":false}
  ],
  "reasoning": "Click input first, then type."
}

EXAMPLE 5 — Key Combo:
{
  "action_id": "step_5",
  "coords_space": "physical",
  "steps": [{"type":"key_combo","combo":["ctrl","s"]}],
  "reasoning": "Save shortcut."
}

EXAMPLE 6 — Scroll:
{
  "action_id": "step_6",
  "coords_space": "physical",
  "steps": [{"type":"scroll","delta":-120,"at":{"x":300,"y":300}}],
  "reasoning": "Scroll down (negative delta)."
}

EXAMPLE 7 — Drag:
{
  "action_id": "step_7",
  "coords_space": "physical",
  "steps": [{"type":"drag","from":{"x":50,"y":50},"to":{"x":500,"y":500},"button":"left"}],
  "reasoning": "Drag from A to B."
}

EXAMPLE 8 — Double-click:
{
  "action_id": "step_8",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":2,"target":{"point":{"x":300,"y":200}}}],
  "reasoning": "Double-click to open."
}

EXAMPLE 9 — Right-click:
{
  "action_id": "step_9",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"right","click_count":1,"target":{"point":{"x":400,"y":300}}}],
  "reasoning": "Right-click for context menu."
}
"""


# Exceptions
class BackendError(Exception): pass
class PlanParseError(BackendError): pass
class SUTCommunicationError(BackendError): pass
class LLMCommunicationError(BackendError): pass

# Data classes
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
    token_usage: Optional[Dict[str, Any]] = None

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

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: str = ""

# ============================================================================
# SIMPLIFIED VALIDATOR
# ============================================================================

class ExpectedResultValidator:
    
    ACTION_VERBS = {
        'closing': ['closes', 'closed', 'disappears', 'gone', 'hide'],
        'loading': ['loads', 'opens', 'appears', 'show'],
        'visibility': ['visible', 'shown', 'displayed'],
        'selection': ['selected', 'highlighted', 'checked', 'active'],
        'state_change': ['starts', 'stops', 'running', 'paused']
    }
    
    def validate(self, expected_result: str, visible_names: List[str], ui_changed: bool, change_magnitude: float) -> Tuple[bool, str]:
        conditions = self._parse_conditions(expected_result)
        
        for i, cond in enumerate(conditions, 1):
            passed, reason = self._validate_single_condition(cond, visible_names, ui_changed, change_magnitude)
            if not passed:
                return (False, f"Condition {i} failed: {reason}")
        
        return (True, "All conditions passed")
    
    def _parse_conditions(self, expected_result: str) -> List[Dict]:
        if " and " in expected_result.lower():
            parts = expected_result.split(" and ", 1)
            return [self._parse_single(p.strip()) for p in parts]
        return [self._parse_single(expected_result)]
    
    def _parse_single(self, text: str) -> Dict:
        text_lower = text.lower()
        cond_type = 'generic'
        
        for ctype, verbs in self.ACTION_VERBS.items():
            if any(v in text_lower for v in verbs):
            #if any(re.search(rf'\b{re.escape(v)}\b', text_lower) for v in verbs):
                cond_type = ctype
                break
        
        # Support both straight quotes ("') and curly/smart quotes (''"")
        quoted = re.findall(r'["\'''"""]([^"\'''"""]+)["\'''"""]', text)
        targets = {item.lower() for item in quoted}
        
        if not targets:
            words = re.findall(r'\b\w{3,}\b', text_lower)
            stopwords = {
                'and', 'or', 'the', 'is', 'with', 'that', 'to', 'of', 'in', 'on', 'at',
                'dialog', 'panel', 'window', 'screen', 'opens', 'loads', 'appears', 'closes', 'shows','visible', 
                'shown', 'displayed', 'selected', 'highlighted', 'checked', 'active'
            }
            targets = set(words) - stopwords
        
        return {'type': cond_type, 'targets': targets}
    
    def _match_targets(self, targets: Set[str], visible_names: List[str]) -> Tuple[int, int]:
        def fuzzy_match(target: str, name: str) -> bool:
            # Normalize both strings
            target_norm = re.sub(r'[^a-z0-9]', '', target.lower())
            name_norm = re.sub(r'[^a-z0-9]', '', name.lower())
            
            # Exact match after normalization
            if target_norm == name_norm:
                return True
            
            if target_norm.isdigit() and len(target_norm) <= 6:
                if target_norm in name_norm:
                    return True
            
            # Fuzzy match with length constraint (OCR tolerance)
            if len(target_norm) >= 3 and len(name_norm) >= 3:
                len_ratio = len(target_norm) / len(name_norm)
                # Tighter constraint: prevents "communicationtest" matching "communicationtestbackup"
                if 0.65<= len_ratio <= 1.35:
                    if SequenceMatcher(None, target_norm, name_norm).ratio() > 0.55:
                        return True
            
            return False
        
        found = sum(1 for t in targets if any(fuzzy_match(t, n) for n in visible_names))
        return found, len(targets)
    
    def _validate_single_condition(self, cond: Dict, visible_names: List[str], ui_changed: bool, change_magnitude: float) -> Tuple[bool, str]:
        ctype = cond['type']
        targets = cond['targets']
        
        if ctype == 'closing':
            found, total = self._match_targets(targets, visible_names)
            
            # 1. Tam Başarı: Hedef kelimelerin hiçbiri yok (Mükemmel durum)
            if found == 0: 
                return (True, "Targets absent")
            
            # 2. Kısmi Başarı : 
            # Ekranda "Scenario" kaldı ama "Management" gitti (found < total).
            # Eğer UI da değiştiyse, demek ki pencere kapandı.
            if ui_changed and found < total:
                return (True, f"Partial removal ({found}/{total} left) + UI Changed")
            
            # 3. Büyük UI Değişimi: Kelimeler hala ekranda görünse bile (OCR hatası olabilir),
            # ekranın %5'inden fazlası değiştiyse kapandığını varsay.
            if change_magnitude > 0.05: 
                 return (True, f"Major UI change ({change_magnitude:.1%}) implies closing")

            return (False, f"{found}/{total} still present")
        
        elif ctype in ['loading', 'visibility']:
            # Açılma işlemleri için UI değişimi + kelime kontrolü
            if ui_changed and change_magnitude > 0.003:  #type  adımları için küçülttüm 
                found, total = self._match_targets(targets, visible_names)
                
                # Ekranda hedef kelimelerden HİÇBİRİ yoksa ama UI çok değiştiyse yine de uyararak geçir
                if total > 0 and found > 0:
                #or change_magnitude > 0.05:
                    return (True, f"UI changed {change_magnitude:.1%} + targets present")
                
                # EXCEPTION: Eğer target yok ise (generic check)
                if total == 0 and change_magnitude > 0.05:
                    return (True, f"Major UI change ({change_magnitude:.1%}), no specific targets")
                # Target var ama bulunamadı
                if total > 0 and found == 0:
                    return (False, f"UI changed but targets '{list(targets)}' not found")
            
            found, total = self._match_targets(targets, visible_names)
            if found >= (total * 0.75):  # 75% gerekli
                return (True, f"{found}/{total} matches (>=75%)")
            else:
                return (False, f"Only {found}/{total} found")
        
        elif ctype == 'selection':
            # Seçim işlemleri: UI değiştiyse (renk değişimi vs.) başarılı say
            if ui_changed:
                found, total = self._match_targets(targets, visible_names)
                if total > 0 and found == 0:
                     return (True, "UI changed (Selection assumed)")
                return (True, "UI changed + target present")
            
            found, total = self._match_targets(targets, visible_names)
            if found > 0: return (True, "Target found")
            return (False, "Target not found")
        
        else:
            found, total = self._match_targets(targets, visible_names)
            if total == 0:
                return (False, "Could not extract any target keywords")

            if found > 0 and found >= (total + 1) // 2:
                return (True, "Generic check passed")
            return (False, f"Low match {found}/{total}")


# ============================================================================
# MAIN BACKEND
# ============================================================================

class LLMBackend:
    
    def __init__(self, state_url_windriver: str, state_url_ods: str, action_url: str, llm_provider: str, llm_model: str, llm_base_url: str = "http://localhost:11434", llm_api_key: Optional[str] = None, *, system_prompt: str = SYSTEM_PROMPT, max_attempts: int = 2, post_action_delay: float = 0.5, sut_timeout: float = 150.0, llm_timeout: float = 800.0, max_tokens: int = 384, max_plan_steps: int = 10, schema_retry_limit: int = 1, http_referrer: str = "https://agentest.local/backend", client_title: str = "AgenTest LLM Backend", enforce_json_response: bool = True, llamacpp_n_ctx: int = 4096, llamacpp_n_gpu_layers: int = 0, llamacpp_n_batch: int = 512) -> None:
        
        self.state_url_windriver = state_url_windriver
        self.state_url_ods = state_url_ods
        self.max_attempts = max_attempts
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
        self.schema_retry_limit = schema_retry_limit
        self.http_referrer = http_referrer
        self.client_title = client_title
        self.llamacpp_n_ctx = llamacpp_n_ctx
        self.llamacpp_n_gpu_layers = llamacpp_n_gpu_layers
        self._llama_model: Optional[Llama] = None
        self.llamacpp_n_batch = llamacpp_n_batch
        
        if llm_provider == "llamacpp" and LLAMACPP_AVAILABLE:
            self._init_llamacpp_model()
        
        self.validator = ExpectedResultValidator()
        self.semantic_filter = SemanticStateFilter(row_tolerance=40, match_threshold=0.70)
        self._use_semantic_filter = True
    
    def _init_llamacpp_model(self) -> None:
        if not os.path.exists(self.model):
            raise FileNotFoundError(f"Model not found: {self.model}")
        
        try:
            self._llama_model = Llama(model_path=self.model, n_ctx=self.llamacpp_n_ctx, n_gpu_layers=self.llamacpp_n_gpu_layers, n_batch=self.llamacpp_n_batch, verbose=False)
            logger.info("Model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load llama.cpp: {e}")
    
    @classmethod
    def from_env(cls, **overrides) -> "LLMBackend":
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
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2): return self._levenshtein_distance(s2, s1)
        if len(s2) == 0: return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                current_row.append(min(previous_row[j + 1] + 1, current_row[j] + 1, previous_row[j] + (c1 != c2)))
            previous_row = current_row
        return previous_row[-1]
    
    def _check_coordinate_element_change(self, clicked_coords: Optional[Tuple[int, int]], state_before: Dict, state_after: Dict, tolerance: int = 5) -> Tuple[bool, str]:
        if not clicked_coords: return (False, "No coords")
        
        click_x, click_y = clicked_coords
        
        elem_before = None
        for elem in state_before.get("elements", []):
            center = elem.get("center", {})
            ex, ey = center.get("x", -999), center.get("y", -999)
            if abs(ex - click_x) <= tolerance and abs(ey - click_y) <= tolerance:
                elem_before = elem
                break
        
        if not elem_before: return (False, "No element at coords (before)")
        
        elem_after = None
        for elem in state_after.get("elements", []):
            center = elem.get("center", {})
            ex, ey = center.get("x", -999), center.get("y", -999)
            if abs(ex - click_x) <= tolerance and abs(ey - click_y) <= tolerance:
                elem_after = elem
                break
        
        if not elem_after: return (True, "Element disappeared")
        
        name_before = elem_before.get("name", "").strip().lower()
        name_after = elem_after.get("name", "").strip().lower()
        
        if name_before != name_after:
            dist = self._levenshtein_distance(name_before, name_after)
            max_len = max(len(name_before), len(name_after))
            if max_len > 0 and (dist / max_len) >= 0.20:
                return (True, f"Name changed ({dist}/{max_len})")
        
        if elem_before.get("type") != elem_after.get("type"): return (True, "Type changed")
        if elem_before.get("is_selected") != elem_after.get("is_selected"): return (True, "Selection changed")
        
        return (False, "No change at coords")
    
    # ========================================================================
    # SCENARIO & STEP EXECUTION 
    # ========================================================================
    
    async def run_scenario(
        self, 
        scenario_name: str, 
        steps: List[StepDefinition], 
        *, 
        temperature: float = 0.1,
        save_recording: bool = True,
        progress_callback: Optional[callable] = None,
        cancel_check: Optional[callable] = None
    ) -> ScenarioResult:
        
        if not steps: raise ValueError("steps required")
        
        # Helper to emit progress
        def emit(msg: str):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)
        
        start_time = time.time()  # Track duration for saving
        
        history = []
        outcomes = []
        final_state = None
        
        # Sadece saf aksiyon planlarını tutacak liste
        raw_plans: List[Dict[str, Any]] = []
        recording_aborted = False
        
        emit(f"STARTING SCENARIO: {scenario_name}")
        
        for idx, step in enumerate(steps, 1):
            # Check for cancellation before each step
            if cancel_check and cancel_check():
                emit("EXECUTION STOPPED BY USER")
                return ScenarioResult("stopped", outcomes, final_state, "User cancelled")
            
            emit(f"STEP {idx}/{len(steps)}: {step.test_step}")
            
            result = await self.run_step(
                step.test_step, 
                step.expected_result, 
                step.note_to_llm, 
                recent_actions=history, 
                temperature=temperature,
                progress_callback=progress_callback,
                cancel_check=cancel_check
            )
            
            outcomes.append(ScenarioStepOutcome(step, result))
            final_state = result.final_state
            
            # --- KAYIT ---
            if result.status == "passed" and result.last_plan:
                raw_plans.append(result.last_plan)
            else:
                recording_aborted = True
                if save_recording:
                    logger.warning(f"Step {idx} failed. Recording aborted.")
            # ---------------------
            
            for log in result.actions:
                history.append(self._summarise_for_prompt(log))
            history = history[-3:]
            
            if result.status != "passed":
                return ScenarioResult(result.status, outcomes, final_state, result.reason)
        
        # Test bitti, birleştirip kaydet
        if save_recording and not recording_aborted and raw_plans:
            # Step tanımlarını da gönder
            step_definitions = [
                {
                    "test_step": s.test_step,
                    "expected_result": s.expected_result,
                    "note_to_llm": s.note_to_llm
                } for s in steps
            ]
            duration = round(time.time() - start_time, 2)
            self._save_merged_recording(scenario_name, raw_plans, step_definitions, outcomes, duration)
        
        return ScenarioResult("passed", outcomes, final_state)
    
    def _save_merged_recording(self, name: str, plans: List[Dict[str, Any]], step_definitions: Optional[List[Dict[str, Any]]] = None, outcomes: Optional[List] = None, duration: float = 0) -> None:
        """
        Tüm adımları TEK BİR Action Payload olarak birleştirip kaydeder.
        Step tanımları ve execution sonuçları da ayrıca kaydedilir.
        """
        try:
            # Dosya ismini temizle
            safe_name = re.sub(r'[^\w\-_\. ]', '', name).replace(' ', '_')
            filename = f"{safe_name}.json"
            directory = "saved_tests"
            
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            filepath = os.path.join(directory, filename)
            
            # --- BİRLEŞTİRME MANTIĞI ---
            merged_steps = []
            
            for i, plan in enumerate(plans):
                # O adıma ait alt adımları (click, type vs) al
                current_steps = plan.get("steps", [])
                
                if not current_steps:
                    continue
                    
                # Listeye ekle
                merged_steps.extend(current_steps)
                
                # Eğer son adım değilse, adımlar arası UI'ın kendine gelmesi için
                # küçük bir 'wait' ekle (Örn: 500ms). Bu, 'Blind Replay' güvenliğini artırır.
                if i < len(plans) - 1:
                    merged_steps.append({"type": "wait", "ms": 1000})
            
            # Execution sonuçlarını kaydet (See Results için)
            execution_result = None
            if outcomes:
                execution_result = []
                for outcome in outcomes:
                    outcome_dict = {
                        "step": {
                            "test_step": outcome.step.test_step,
                            "expected_result": outcome.step.expected_result,
                            "note_to_llm": outcome.step.note_to_llm
                        },
                        "result": {
                            "status": outcome.result.status,
                            "attempts": outcome.result.attempts,
                            "reason": outcome.result.reason,
                            "actions": []
                        }
                    }
                    # Actions'ı serileştir
                    for action in outcome.result.actions:
                        action_dict = {
                            "action_id": action.action_id,
                            "plan": action.plan,
                            "ack": action.ack,
                            "state_before": action.state_before,
                            "state_after": action.state_after
                        }
                        outcome_dict["result"]["actions"].append(action_dict)
                    execution_result.append(outcome_dict)
            
            # Action Endpoint'in beklediği formatı oluştur + step tanımları + execution result
            final_payload = {
                "action_id": f"replay_{safe_name}_{int(time.time())}",
                "coords_space": "physical",
                "steps": merged_steps,
                "step_definitions": step_definitions or [],  # Orijinal test adımları
                "execution_result": execution_result,  # İlk çalıştırma sonuçları
                "execution_duration": duration  # Orijinal çalıştırma süresi
            }
            # ---------------------------
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_payload, f, indent=2, ensure_ascii=False)
                
            logger.info("MERGED SCENARIO RECORDED: %s", filepath)
            
        except Exception as e:
            logger.error("Failed to save merged recording: %s", e)

    async def run_step(self, test_step: str, expected_result: str, note_to_llm: Optional[str] = None, *, recent_actions=None, temperature=0.1, progress_callback: Optional[callable] = None, cancel_check: Optional[callable] = None) -> RunResult:
        
        # Helper to emit progress
        def emit(msg: str):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)
        
        actions_log = []
        state = {}
        
        for attempt in range(1, self.max_attempts + 1):
            # Check for cancellation at each attempt
            if cancel_check and cancel_check():
                emit("EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            
            method = "WinDriver" if attempt == 1 else "ODS"
            emit(f"Attempt {attempt}/{self.max_attempts} ({method})")
            
            state = await self._fetch_state(attempt)

            # Skip to ODS if note_to_llm contains [ODS] tag
            if attempt == 1 and note_to_llm and "[ODS]" in note_to_llm.upper():
                emit("SKIP TO ODS: [ODS] tag found in note_to_llm")
                continue

            if attempt == 1 and "elements" in state:
                target_match = re.search(r"'([^']+)'", test_step)
                if target_match:
                    target_name = target_match.group(1).lower().strip()
                 
                    # Aranan kelime jenerik değilse (örn: 'Save', 'General') kontrol et
                    #if target_name not in generic_terms and len(target_name) > 2:
                    if target_name and len(target_name) > 2:
                        found = any(target_name == (e.get("name", "") or "").lower().strip() for e in state.get("elements", []))
                        
                        if not found:
                            emit(f"PRE-CHECK: '{target_name}' NOT found -> ODS")
                            continue
            
            # Plan
            emit("SENDING TO LLM...")
            
            # Check for cancellation before LLM request
            if cancel_check and cancel_check():
                emit("EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            
            try:
                # Create task for LLM request so we can poll for cancellation
                llm_task = asyncio.create_task(
                    self._request_plan(
                        test_step=test_step, 
                        expected_result=expected_result, 
                        note_to_llm=note_to_llm, 
                        state=state, 
                        recent_actions=recent_actions or [], 
                        temperature=temperature, 
                        attempt_number=attempt
                    )
                )
                
                # Poll for cancellation while waiting for LLM
                while not llm_task.done():
                    if cancel_check and cancel_check():
                        llm_task.cancel()
                        emit("EXECUTION STOPPED BY USER")
                        return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
                    await asyncio.sleep(0.5)  # Check every 500ms
                
                plan, llm_view, token_usage = await llm_task
                
                # Emit token usage to UI
                if token_usage and token_usage.total_tokens > 0:
                    emit(f"TOKEN USAGE [{token_usage.provider}]: Input={token_usage.input_tokens}, Output={token_usage.output_tokens}, Total={token_usage.total_tokens}")
            except asyncio.CancelledError:
                emit("EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            except Exception as e:
                emit(f"Planning failed: {e}")
                continue
            
            # Check for cancellation after LLM request
            if cancel_check and cancel_check():
                emit("EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            
            steps = plan.get("steps", [])
            
            if not steps:
                if attempt < self.max_attempts:
                    continue
                return RunResult("failed", attempt, actions_log, state, plan, "No actions")
            
            emit(f"PARSED PLAN: {len(steps)} action(s)")
            
            # Execute
            emit("EXECUTING ACTION...")
            state_before = state  # Keep original state for validation
            
            # Check for cancellation before action
            if cancel_check and cancel_check():
                emit(" EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            
            try:
                ack = await self._send_action(plan)
            except Exception as e:
                emit(f"Action failed: {e}")
                continue
            
            emit("VALIDATION CHECK...")
            final_state = await self._fetch_state_safe(state, attempt)
            
            # Add llm_view to state_before copy for UI display only
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
            
            log = ActionExecutionLog(plan.get("action_id", ""), plan, ack, state_before_with_llm, final_state, token_usage=token_usage_dict)
            actions_log.append(log)
            
            # Validate using original state_before (without llm_view)
            if self._expected_holds(final_state, expected_result, plan, state_before, test_step):
                emit("PASSED")
                return RunResult("passed", attempt, actions_log, final_state, plan, f"Success ({method})")
        
        emit("STEP FAILED")
        return RunResult("failed", self.max_attempts, actions_log, state, None, "Max attempts")
    
    async def _fetch_state(self, attempt: int = 1) -> Dict:
        url = self.state_url_ods if attempt == 2 else self.state_url_windriver
        async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
            resp = await client.post(url, json={})
            resp.raise_for_status()
            return resp.json()
    
    async def _fetch_state_safe(self, fallback: Dict, attempt: int = 1) -> Dict:
        try:
            return await self._fetch_state(attempt)
        except:
            return fallback
    
    async def _send_action(self, plan: Dict) -> Dict:
        async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
            resp = await client.post(self.action_url, json=plan)
            resp.raise_for_status()
            return resp.json()
    
    def _detect_ui_change(self, before: Dict, after: Dict) -> Tuple[bool, float]:
        def get_hash(s):
            items = sorted([f"{e.get('name')}@{e.get('center')}" for e in s.get("elements", [])])
            return hashlib.md5("|".join(items).encode()).hexdigest()
        
        if get_hash(before) == get_hash(after): return False, 0.0
        
        c1 = len(before.get("elements", []))
        c2 = len(after.get("elements", []))
        ratio = abs(c1 - c2) / c1 if c1 > 0 else 1.0
         # FIXED: If hash differs but count is same, calculate actual change magnitude
        if c1 == c2:
            # Count how many elements actually changed
            before_items = {f"{e.get('name')}@{e.get('center')}" for e in before.get("elements", [])}
            after_items = {f"{e.get('name')}@{e.get('center')}" for e in after.get("elements", [])}

            changed = len(before_items.symmetric_difference(after_items))
            ratio = changed / c1 if c1 > 0 else 0.0
        return True, ratio

    # ========================================================================
    # VALIDATION 
    # ========================================================================
    
    def _expected_holds(self, state: Dict, expected_result: str, plan: Dict, state_before: Optional[Dict] = None, test_step: Optional[str] = None) -> bool:
        
        logger.info("=" * 80)
        logger.info("VALIDATION CHECK")
        logger.info("=" * 80)
        logger.info("Expected result: %s", expected_result)
        
        if not expected_result:
            logger.warning("No expected result provided")
            return False
        
        steps = plan.get("steps", [])
        if not steps:
            logger.warning("No steps in plan")
            return False
        
        # Get coords from last click
        clicked_coords = None
        for step in reversed(steps):
            if step.get("type") == "click":
                pt = step.get("target", {}).get("point", {})
                if "x" in pt: clicked_coords = (int(pt["x"]), int(pt["y"]))
                break
        
        if clicked_coords:
            logger.info("Clicked coordinates: (%d, %d)", clicked_coords[0], clicked_coords[1])
        else:
            logger.info("No click coordinates (keyboard action or other)")
        
        elems = state.get("elements", [])
        if not elems:
            logger.warning("No elements in state")
            return False
        
        logger.info("Elements in current state: %d", len(elems))
        
        visible_names = [e.get("name", "").lower() for e in elems if e.get("name")]
        logger.info("Visible element names: %d unique names", len(set(visible_names)))
        
        ui_changed, change_mag = self._detect_ui_change(state_before, state) if state_before else (False, 0.0)
        logger.info("UI changed: %s (magnitude: %.1f%%)", "YES" if ui_changed else "NO", change_mag * 100)
        
        coord_changed, coord_reason = (False, "")

        if clicked_coords and state_before:
            coord_changed, coord_reason = self._check_coordinate_element_change(clicked_coords, state_before, state)
        
        if coord_changed:
            logger.info("Coordinate element changed: YES (%s)", coord_reason)
        else:
            logger.info("Coordinate element changed: NO")
        
        expected_lower = expected_result.lower()

        is_selection = any(v in expected_lower for v in ['selected', 'highlighted', 'checked'])
        is_closing_action = any(v in expected_lower for v in ['closes', 'gone', 'hide'])
        
        logger.info("Is selection action: %s", is_selection)
        logger.info("Is closing action: %s", is_closing_action)

        if coord_changed and (is_selection or is_closing_action):
            logger.info("PRIORITY PASS: Coordinate change detected (%s)", coord_reason)
            logger.info("=" * 80)
            return True
        
        logger.info("-" * 80)
        logger.info("Running standard validation...")
        logger.info("-" * 80)
        
        # Standard validation (UI change + target matching)
        passed, reason = self.validator.validate(expected_result, visible_names, ui_changed, change_mag)
        
        if passed:
            logger.info("VALIDATION PASSED: %s", reason)
        else:
            logger.error("VALIDATION FAILED: %s", reason)
        
        logger.info("=" * 80)
        
        return passed
    
    # LLM Communication
    async def _request_plan(self, *, test_step: str, expected_result: str, note_to_llm: Optional[str], state: Dict, recent_actions: List, temperature: float, attempt_number: int) -> tuple[Dict, str, TokenUsage]:
        
        messages, llm_view = self._build_messages(test_step, expected_result, note_to_llm, state, recent_actions, attempt_number)
        token_usage = TokenUsage()
        
        if self.llm_provider == "local":
            # Local provider uses Ollama API with LOCAL_SYSTEM_PROMPT
            plan, token_usage = await self._request_plan_ollama(messages, temperature)
        elif self.llm_provider == "ollama":
            plan, token_usage = await self._request_plan_ollama(messages, temperature)
        elif self.llm_provider == "llamacpp":
            plan = await self._request_plan_llamacpp(messages, temperature)
        elif self.llm_provider == "openai":
            plan = await self._request_plan_openai(messages, temperature)
        elif self.llm_provider == "lmstudio":
            # LM Studio uses OpenAI-compatible API
            plan = await self._request_plan_openai(messages, temperature)
        elif self.llm_provider == "anthropic":
            plan, token_usage = await self._request_plan_anthropic(messages, temperature)
        elif self.llm_provider == "gemini":
            plan = await self._request_plan_gemini(messages, temperature)
        elif self.llm_provider == "custom":
            # Custom provider uses OpenAI-compatible API
            plan = await self._request_plan_openai(messages, temperature)
        else:
            plan = await self._request_plan_openrouter(messages, temperature)
        
        return plan, llm_view, token_usage
    
    async def _request_plan_ollama(self, messages: List, temperature: float) -> Tuple[Dict, TokenUsage]:
        """Ollama API with JSON mode."""
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            resp = await client.post(f"{self.llm_base_url}/api/chat", json={"model": self.model, "messages": messages, "stream": False, "format": "json", "options": {"temperature": temperature}})
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
        if not self._llama_model: raise LLMCommunicationError("Model not loaded")
        
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: self._llama_model.create_completion(prompt, max_tokens=self.max_tokens, temperature=temperature))
        return self._parse_plan(resp["choices"][0]["text"])
    
    async def _request_plan_openai(self, messages: List, temperature: float) -> Dict:
        """OpenAI-compatible API with JSON mode."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"}  # Force JSON output
        }
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            resp = await client.post(f"{self.llm_base_url}/chat/completions", headers=headers, json=payload)
            return self._parse_plan(resp.json()["choices"][0]["message"]["content"])
    
    async def _request_plan_openrouter(self, messages: List, temperature: float) -> Dict:
        """OpenRouter API with JSON mode."""
        headers = {"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": self.http_referrer}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"}  # Force JSON output
        }
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            return self._parse_plan(resp.json()["choices"][0]["message"]["content"])
    
    async def _request_plan_anthropic(self, messages: List, temperature: float) -> Tuple[Dict, TokenUsage]:
        """Anthropic Claude API support."""
        if not ANTHROPIC_AVAILABLE:
            raise LLMCommunicationError("anthropic package not installed")
        
        # Extract system prompt and user message
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"]
        
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
        
        # Build contents with system instruction and JSON schema enforcement
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.model,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=sys_msg,
                    temperature=temperature,
                    # Set minimum 8192 to ensure enough budget for thinking + output
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
            
            # Check content parts
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for i, part in enumerate(candidate.content.parts):
                        part_text = getattr(part, 'text', str(part))
                        logger.info(f"  Part {i}: {part_text[:200]}...")
        else:
            logger.warning("  No candidates in response!")
            logger.info(f"  Full response: {response}")
        
        # Extract text content from response
        content = response.text if response.text else ""
        logger.info(f"  response.text length: {len(content)}")
        logger.info(f"  response.text preview: {content[:300]}...")
        
        if not content or len(content.strip()) < 10:
            raise LLMCommunicationError("Gemini returned empty/incomplete response. Check finish_reason above.")
        
        return self._parse_plan(content)
    
    def _build_messages(self, test_step: str, expected_result: str, note_to_llm: Optional[str], state: Dict, recent_actions: List, attempt_number: int) -> tuple[List[Dict], str]:
        
        elements_to_show = state.get("elements", [])
        
        # 1. Semantic Filter Uygula (Eğer aktifse)
        if self._use_semantic_filter and elements_to_show:
            filtered = self.semantic_filter.filter_elements(elements_to_show, test_step, expected_result, note_to_llm or "")

            # DEBUG: Filtreleme istatistikleri
            logger.info("=" * 80)
            logger.info("SEMANTIC FILTER STATS")
            logger.info("=" * 80)
            logger.info("Original elements: %d", len(elements_to_show))
            logger.info("Filtered elements: %d", len(filtered))
            
            elements_to_show = filtered

        # 2. LLM View Oluştur
        
        lines = []
        lines.append("Format: (x,y) [Type] Name | Value | AutomationID")
        lines.append("-" * 60)
        
        for el in elements_to_show:
            name = el.get("name", "NoName").replace("|", "")
            e_type = el.get("type", "unknown")
            center = el.get("center", {})
            x, y = center.get("x", 0), center.get("y", 0)
            value = el.get("value", "")  # Value alanı eklenmeli
            automation_id = el.get("automation_id", "")  # ID alanı eklenmeli
    
            # Format: (x,y) [Type] Name | Value | AutomationID
            value_str = f"Value: {value}" if value else ""
            id_str = f"ID: {automation_id}" if automation_id else ""
    
            lines.append(f"({x},{y}) [{e_type}] {name} | {value_str} | {id_str}")
            
        llm_view = "\n".join(lines)
        
        # DEBUG LOG: LLM'e giden nihai görüntü
        logger.info("=" * 80)
        logger.info("SENDING TO LLM")
        logger.info("=" * 80)
        logger.info("Test step: %s", test_step)
        logger.info("Expected result: %s", expected_result)
        logger.info("Note to LLM: %s", note_to_llm)
        logger.info("=" * 80)
        logger.info("Current View:\n%s", llm_view)
        logger.info("=" * 80)
        
        payload = {"test_step": test_step, "expected_result": expected_result, "current_state": llm_view, "note_to_llm": note_to_llm}
        
        if recent_actions:
            payload["recent_actions"] = recent_actions[-2:]
        
        if attempt_number > 1:
            payload["retry_context"] = {"attempt": attempt_number, "method": "ODS"}
        
        # Select prompt based on provider type
        # "local" provider uses LOCAL_SYSTEM_PROMPT (optimized for Qwen)
        # All other providers use SYSTEM_PROMPT
        if self.llm_provider == "local":
            active_prompt = LOCAL_SYSTEM_PROMPT.strip()
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
        logger.info("RECEIVED FROM LLM")
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
            plan["action_id"] = f"step_{int(time.time())}"
        if "coords_space" not in plan:
            plan["coords_space"] = "physical"
        
        # DEBUG LOG: Parsed plan
        logger.info("=" * 80)
        logger.info("PARSED PLAN")
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
    
    def _validate_plan_against_screen(self, plan: Dict, state: Dict) -> Optional[str]:
        screen = state.get("screen", {})
        sw, sh = screen.get("w", 0), screen.get("h", 0)
        if sw <= 0 or sh <= 0: return None
        
        for idx, step in enumerate(plan.get("steps", [])):
            if step.get("type") == "click":
                pt = step.get("target", {}).get("point", {})
                x, y = pt.get("x", -1), pt.get("y", -1)
                if not (0 <= x < sw and 0 <= y < sh):
                    return f"step[{idx}] out of bounds"
        return None
    
    def _summarise_for_prompt(self, entry: ActionExecutionLog) -> Dict:
        return {"action_id": entry.action_id, "steps_count": len(entry.plan.get("steps", [])), "status": entry.ack.get("status")}
    
# ============================================================================
# RAW ODS BACKEND (FOR DEMONSTRATION / COMPARISON ONLY)
# Purpose: To show how the LLM fails or struggles without semantic filtering,
#          WinDriver hybrid fetch, and pre-checks.
# ============================================================================

class RawODSBackend(LLMBackend):
    """
    A 'Naive' implementation that strictly fetches from ODS and sends 
    the raw (unfiltered) element list to the LLM.
    Used to demonstrate the value of the optimizations in the main LLMBackend.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_semantic_filter = False 
        self.max_attempts = 1
    
    async def _fetch_state(self, attempt: int = 1) -> Dict:
        """Override to ALWAYS use ODS URL, ignoring attempt number/logic."""
        # Force ODS URL
        url = self.state_url_ods
        logger.info(" Fetching RAW State from ODS: %s", url)
        
        async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
            resp = await client.post(url, json={})
            resp.raise_for_status()
            return resp.json()

    async def run_step(self, test_step: str, expected_result: str, note_to_llm: Optional[str] = None, *, recent_actions=None, temperature=0.1) -> RunResult:
        """
        Simplified execution flow:
        1. Fetch ODS State
        2. Send ALL elements to LLM (No Filter)
        3. Execute Action
        4. Validate
        (No retries, no WinDriver fallbacks, no Pre-checks)
        """
        actions_log = []
        state = {}
        
        logger.warning("RUNNING IN RAW ODS MODE (Benchmark Mode)")
        logger.warning("   - No WinDriver Fallback")
        logger.warning("   - No Semantic Filtering")
        logger.warning("   - No Pre-checks")

        # 1. Fetch State (ODS Only)
        try:
            state = await self._fetch_state()
        except Exception as e:
            logger.error("ODS Fetch Failed: %s", e)
            return RunResult("failed", 1, [], {}, None, f"ODS Error: {e}")

        # 2. Plan (Base class will see self._use_semantic_filter=False and send everything)
        try:
            plan, llm_view = await self._request_plan(
                test_step=test_step, 
                expected_result=expected_result, 
                note_to_llm=note_to_llm, 
                state=state, 
                recent_actions=recent_actions or [], 
                temperature=temperature, 
                attempt_number=1
            )
        except Exception as e:
            logger.error("Planning Failed: %s", e)
            return RunResult("failed", 1, [], state, None, f"LLM Error: {e}")

        # Check if plan is empty
        steps = plan.get("steps", [])
        if not steps:
            return RunResult("failed", 1, actions_log, state, plan, "LLM returned no actions")

        # 3. Execute
        state_before = state  # Keep original state for validation
        try:
            ack = await self._send_action(plan)
        except Exception as e:
            logger.error("Action Execution Failed: %s", e)
            return RunResult("failed", 1, [], state, plan, f"Action Error: {e}")

        # 4. Fetch Result State (For validation)
        final_state = await self._fetch_state_safe(state, 1)
        
        # Add llm_view to state_before copy for UI display only
        state_before_with_llm = {**state_before, "llm_view": llm_view}
        log = ActionExecutionLog(plan.get("action_id", ""), plan, ack, state_before_with_llm, final_state)
        actions_log.append(log)

        # 5. Validate using original state_before (without llm_view)
        if self._expected_holds(final_state, expected_result, plan, state_before, test_step):
            logger.info("RAW MODE SUCCESS (Surprisingly!)")
            return RunResult("passed", 1, actions_log, final_state, plan, "Success (Raw ODS)")
        
        logger.info("RAW MODE FAILED (As expected?)")
        return RunResult("failed", 1, actions_log, final_state, plan, "Validation Failed")
