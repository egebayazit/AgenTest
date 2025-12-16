# llm_backend.py v2.7 - Simplified Dialog Validation + DEBUG LOGGING
# KEY CHANGES from v2.4:
# 1. Opening actions ("opens", "loads") SKIP dialog bounds entirely  
# 3. Selection = any UI change (no coordinate requirement)
# 4. Removed complex dialog detection logic
# 5. Simplified validation - focus on UI change detection
# 6. ADDED: Comprehensive debug logging for LLM input/output

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

# Import SemanticStateFilter (assumed to be in same module or package)
from semantic_filter import SemanticStateFilter

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a Windows UI automation expert. Execute test steps using ONLY elements from current_state.

OUTPUT: Valid JSON only. No markdown, no explanations.

CURRENT_STATE FORMAT:
ID | Name | Type | (x,y)
------------------------------------------------------------
1 | Save Settings | text | (10,20)
2 | Trash Bin | icon | (100,200)


CORE PRINCIPLE:
1. SPATIAL IS KING: If user says "next to", "near", "right of" -> PROXIMITY is PRIMARY.
   - Find Anchor -> Find closest element in direction -> Click that (Ignore name if needed).
2. If NO spatial words -> Use exact NAME match.

ACTION RULES (STRICT):
- ACTION_SEQUENCE_ALLOWED: If a test_step requires multiple actions (e.g., click + type), produce all actions in correct logical order.
- If note_to_llm specifies an order (e.g. 'click before typing'), follow that strictly.
1. 'click': Requires "target": {"point": {"x":..., "y":...}}.
   - Optional: "modifiers": ["ctrl", "shift"] for multi-select.
   - Optional: "button": "right" for context menus.
2. 'type': Use for text/number entry.
   - Required: "text": "string".
   - Optional: "enter": true (to submit).
3. 'key_combo': Use for shortcuts ONLY.
   - Format: "combo": ["ctrl", "c"] or ["enter"] or ["tab"].
4. 'drag': Move element from A to B.
   - Required: "from": {"x":..., "y":...}, "to": {"x":..., "y":...}.
5. 'scroll': Scroll list/map.
   - Required: "delta": integer (Positive=Up, Negative=Down).
   - Optional: "at": {"x":..., "y":...} (Mouse position during scroll).
6. 'wait': Pause execution.
   - Required: "ms": integer (milliseconds).

EXAMPLES:

Input: Click dropdown next to 'Force Id' (Anchor 'Force Id' at 100,100. 'Move Down' icon at 150,100)
Output:
{
  "action_id": "step_1",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":1,"target":{"point":{"x":150,"y":100}}}],
  "reasoning": "Found anchor 'Force Id'. Closest dropdown-like element is 'Move Down' at (150,100)."
}
EXAMPLES:

EXAMPLE 1 — Click (spatial):
{
  "action_id": "step_1",
  "coords_space": "physical",
  "steps": [
    {"type":"click","button":"left","click_count":1,
     "target":{"point":{"x":150,"y":100}}}
  ],
  "reasoning":"Anchor located; nearest matching element clicked."
}

EXAMPLE 2 — Type:
{
  "action_id": "step_2",
  "coords_space": "physical",
  "steps": [
    {"type":"type","text":"Hello","enter":true}
  ],
  "reasoning":"Typed text and submitted with Enter."
}

EXAMPLE 3 — Drag:
{
  "action_id":"step_3",
  "coords_space":"physical",
  "steps":[
    {"type":"drag","from":{"x":50,"y":50},
     "to":{"x":500,"y":500},"button":"left"}
  ],
  "reasoning":"Dragged source to target coordinates."
}

EXAMPLE 4 — Scroll:
{
  "action_id":"step_4",
  "coords_space":"physical",
  "steps":[
    {"type":"scroll","delta":-120,"at":{"x":300,"y":300}}
  ],
  "reasoning":"Scrolling downward."
}

EXAMPLE 5 — Key Combo (CTRL+C):
{
  "action_id":"step_5",
  "coords_space":"physical",
  "steps":[
    {"type":"key_combo","combo":["ctrl","c"]}
  ],
  "reasoning":"Triggered copy shortcut."
}

EXAMPLE 6 — Key Combo (ALT+F4):
{
  "action_id":"step_6",
  "coords_space":"physical",
  "steps":[
    {"type":"key_combo","combo":["alt","f4"]}
  ],
  "reasoning":"Triggered window close shortcut."
}

EXAMPLE 7 — Key Combo (ENTER):
{
  "action_id":"step_7",
  "coords_space":"physical",
  "steps":[
    {"type":"key_combo","combo":["enter"]}
  ],
  "reasoning":"Pressed Enter key."
}

"""


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
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "oneOf": [
                    {"properties": {"type": {"const": "click"}, "button": {"enum": ["left", "right", "middle"]}, "click_count": {"type": "integer", "minimum": 1, "maximum": 2}, "modifiers": {"type": "array", "items": {"enum": ["ctrl", "shift", "alt", "win"]}}, "target": {"type": "object", "properties": {"point": {"type": "object", "required": ["x", "y"], "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}}, "required": ["point"]}}, "required": ["type", "button", "click_count", "target"]},
                    {"properties": {"type": {"const": "type"}, "text": {"type": "string"}, "delay_ms": {"type": "integer"}, "enter": {"type": "boolean"}}, "required": ["type", "text"]},
                    {"properties": {"type": {"const": "key_combo"}, "combo": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["type", "combo"]},
                    {"properties": {"type": {"const": "wait"}, "ms": {"type": "integer"}}, "required": ["type", "ms"]},
                    {"properties": {"type": {"const": "drag"}, "from": {"type": "object", "required": ["x", "y"]}, "to": {"type": "object", "required": ["x", "y"]}, "button": {"enum": ["left", "right", "middle"]}}, "required": ["type", "from", "to"]},
                    {"properties": {"type": {"const": "scroll"}, "delta": {"type": "integer"}, "horizontal": {"type": "boolean"}, "at": {"type": "object", "required": ["x", "y"]}}, "required": ["type", "delta"]},
                ],
            },
        },
    },
}

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
        
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
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
                
                # ⚠️ EXCEPTION: Eğer target yok ise (generic check)
                if total == 0 and change_magnitude > 0.05:
                    return (True, f"Major UI change ({change_magnitude:.1%}), no specific targets")
                # ❌ Target var ama bulunamadı
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
            logger.info("✓ Model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load llama.cpp: {e}")
    
    @classmethod
    def from_env(cls, **overrides) -> "LLMBackend":
        env = os.getenv
        return cls(
            state_url_windriver=overrides.get("state_url_windriver") or env("SUT_STATE_URL_WINDRIVER", "http://127.0.0.1:18800/state/for-llm"),
            state_url_ods=overrides.get("state_url_ods") or env("SUT_STATE_URL_ODS", "http://127.0.0.1:18800/state/from-ods"),
            action_url=overrides.get("action_url") or env("SUT_ACTION_URL", "http://192.168.137.249:18080/action"),
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
    # SCENARIO & STEP EXECUTION (GÜNCELLENDİ)
    # ========================================================================
    
# ========================================================================
    # SCENARIO & STEP EXECUTION (MERGED RECORDING MODU)
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
        
        emit(f"🎬 STARTING SCENARIO: {scenario_name}")
        
        for idx, step in enumerate(steps, 1):
            # Check for cancellation before each step
            if cancel_check and cancel_check():
                emit("🛑 EXECUTION STOPPED BY USER")
                return ScenarioResult("stopped", outcomes, final_state, "User cancelled")
            
            emit(f"📍 STEP {idx}/{len(steps)}: {step.test_step}")
            
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
            
            # --- KAYIT MANTIĞI ---
            if result.status == "passed" and result.last_plan:
                # Wrapper yok, direkt ham planı ekle
                raw_plans.append(result.last_plan)
            else:
                recording_aborted = True
                if save_recording:
                    logger.warning(f"🛑 Step {idx} failed. Recording aborted.")
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
                
            logger.info("💾 MERGED SCENARIO RECORDED: %s", filepath)
            
        except Exception as e:
            logger.error("⚠️ Failed to save merged recording: %s", e)

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
                emit("🛑 EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            
            method = "WinDriver" if attempt == 1 else "ODS"
            emit(f"🔄 Attempt {attempt}/{self.max_attempts} ({method})")
            
            state = await self._fetch_state(attempt)
            
            """# PRE-CHECK
            if attempt == 1:
                target_match = re.search(r"'([^']+)'", test_step)
                if target_match:
                    target = target_match.group(1).lower()
                    if not any(target in (e.get("name") or "").lower() for e in state.get("elements", [])):
                        logger.info("🚀 PRE-CHECK: Target not in WinDriver")
                        continue"""
            

            if attempt == 1 and "elements" in state:
                target_match = re.search(r"'([^']+)'", test_step)
                if target_match:
                    target_name = target_match.group(1).lower().strip()
                    
                    """# Bu kelimeler isimde tam yazmasa bile (örn: type='button') LLM bulabilir.
                    # O yüzden bunları PRE-CHECK ile engellemiyoruz.
                    generic_terms = {
                        "checkbox", "button", "icon", "input", "toggle", "tab", 
                        "panel", "window", "list", "menu", "dropdown", "select", "box",
                        "text", "link", "image", "field"
                    }"""
                    
                    # Aranan kelime jenerik değilse (örn: 'Save', 'General') kontrol et
                    #if target_name not in generic_terms and len(target_name) > 2:
                    if target_name and len(target_name) > 2:
                        found = any(target_name == (e.get("name", "") or "").lower().strip() for e in state.get("elements", []))
                        
                        if not found:
                            emit(f"🚀 PRE-CHECK: '{target_name}' NOT found → ODS")
                            continue
            
            # Plan
            emit("📤 SENDING TO LLM...")
            
            # Check for cancellation before LLM request
            if cancel_check and cancel_check():
                emit("🛑 EXECUTION STOPPED BY USER")
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
                        emit("🛑 EXECUTION STOPPED BY USER")
                        return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
                    await asyncio.sleep(0.5)  # Check every 500ms
                
                plan, llm_view = await llm_task
            except asyncio.CancelledError:
                emit("🛑 EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            except Exception as e:
                emit(f"❌ Planning failed: {e}")
                continue
            
            # Check for cancellation after LLM request
            if cancel_check and cancel_check():
                emit("🛑 EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            
            steps = plan.get("steps", [])
            
            if not steps:
                if attempt < self.max_attempts:
                    continue
                return RunResult("failed", attempt, actions_log, state, plan, "No actions")
            
            emit(f"🎯 PARSED PLAN: {len(steps)} action(s)")
            
            # Execute
            emit("⚡ EXECUTING ACTION...")
            state_before = state  # Keep original state for validation
            
            # Check for cancellation before action
            if cancel_check and cancel_check():
                emit("🛑 EXECUTION STOPPED BY USER")
                return RunResult("stopped", attempt, actions_log, state, None, "User cancelled")
            
            try:
                ack = await self._send_action(plan)
            except Exception as e:
                emit(f"❌ Action failed: {e}")
                continue
            
            emit("🔍 VALIDATION CHECK...")
            final_state = await self._fetch_state_safe(state, attempt)
            
            # Add llm_view to state_before copy for UI display only
            state_before_with_llm = {**state_before, "llm_view": llm_view}
            log = ActionExecutionLog(plan.get("action_id", ""), plan, ack, state_before_with_llm, final_state)
            actions_log.append(log)
            
            # Validate using original state_before (without llm_view)
            if self._expected_holds(final_state, expected_result, plan, state_before, test_step):
                emit("✅ PASSED")
                return RunResult("passed", attempt, actions_log, final_state, plan, f"Success ({method})")
        
        emit("❌ STEP FAILED")
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
    # SIMPLIFIED VALIDATION (NO DIALOG BOUNDS COMPLEXITY) + DEBUG LOGGING
    # ========================================================================
    
    def _expected_holds(self, state: Dict, expected_result: str, plan: Dict, state_before: Optional[Dict] = None, test_step: Optional[str] = None) -> bool:
        
        logger.info("=" * 80)
        logger.info("VALIDATION CHECK")
        logger.info("=" * 80)
        logger.info("Expected result: %s", expected_result)
        
        if not expected_result:
            logger.warning("⚠️ No expected result provided")
            return False
        
        steps = plan.get("steps", [])
        if not steps:
            logger.warning("⚠️ No steps in plan")
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
            logger.warning("⚠️ No elements in state")
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
            logger.info("✅ PRIORITY PASS: Coordinate change detected (%s)", coord_reason)
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
            logger.error("❌ VALIDATION FAILED: %s", reason)
        
        logger.info("=" * 80)
        
        return passed
    
    # LLM Communication
    async def _request_plan(self, *, test_step: str, expected_result: str, note_to_llm: Optional[str], state: Dict, recent_actions: List, temperature: float, attempt_number: int) -> tuple[Dict, str]:
        
        messages, llm_view = self._build_messages(test_step, expected_result, note_to_llm, state, recent_actions, attempt_number)
        
        if self.llm_provider == "ollama":
            plan = await self._request_plan_ollama(messages, temperature)
        elif self.llm_provider == "llamacpp":
            plan = await self._request_plan_llamacpp(messages, temperature)
        elif self.llm_provider == "openai":
            plan = await self._request_plan_openai(messages, temperature)
        else:
            plan = await self._request_plan_openrouter(messages, temperature)
        
        return plan, llm_view
    
    async def _request_plan_ollama(self, messages: List, temperature: float) -> Dict:
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            #resp = await client.post(f"{self.llm_base_url}/api/chat", json={"model": self.model, "messages": messages, "stream": False, "format": AGEN_TEST_PLAN_SCHEMA, "options": {"temperature": temperature}})
            resp = await client.post(f"{self.llm_base_url}/api/chat", json={"model": self.model, "messages": messages, "stream": False, "format": "json", "options": {"temperature": temperature}})
            return self._parse_plan(resp.json()["message"]["content"])
    
    async def _request_plan_llamacpp(self, messages: List, temperature: float) -> Dict:
        if not self._llama_model: raise LLMCommunicationError("Model not loaded")
        
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
        prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: self._llama_model.create_completion(prompt, max_tokens=self.max_tokens, temperature=temperature))
        return self._parse_plan(resp["choices"][0]["text"])
    
    async def _request_plan_openai(self, messages: List, temperature: float) -> Dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            resp = await client.post(f"{self.llm_base_url}/chat/completions", headers=headers, json={"model": self.model, "messages": messages, "temperature": temperature})
            return self._parse_plan(resp.json()["choices"][0]["message"]["content"])
    
    async def _request_plan_openrouter(self, messages: List, temperature: float) -> Dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": self.http_referrer}
        async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={"model": self.model, "messages": messages, "temperature": temperature})
            return self._parse_plan(resp.json()["choices"][0]["message"]["content"])
    
    def _build_messages(self, test_step: str, expected_result: str, note_to_llm: Optional[str], state: Dict, recent_actions: List, attempt_number: int) -> tuple[List[Dict], str]:
        
        elements_to_show = state.get("elements", [])
        
        # 1. Semantic Filter Uygula (Eğer aktifse)
        if self._use_semantic_filter and elements_to_show:
            filtered = self.semantic_filter.filter_elements(elements_to_show, test_step, expected_result, note_to_llm or "")

            # DEBUG: Filtreleme istatistikleri
            logger.info("=" * 80)
            logger.info("📊 SEMANTIC FILTER STATS")
            logger.info("=" * 80)
            logger.info("Original elements: %d", len(elements_to_show))
            logger.info("Filtered elements: %d", len(filtered))
            
            elements_to_show = filtered

        # 2. LLM View Oluştur (MANUEL FORMATLAMA - SYSTEM PROMPT İLE UYUMLU)
        # Hata buradaydı: state.get("llm_view") kullanıldığı için format bozuluyordu.
        # Şimdi manuel olarak ID | Name | Type | (x,y) formatında örüyoruz.
        
        lines = []
        lines.append("ID | Name | Type | (x,y)")
        lines.append("-" * 60)
        
        for idx, el in enumerate(elements_to_show, 1):
            e_id = idx  # 1'den başlayan sanal ID
            """name = str(el.get("name") or "Unknown").replace("|", "")[:40]
            e_type = str(el.get("type") or "unk")[:10] """  #or geldi , str gitti
            name = el.get("name", "Unknown").replace("|", "")[:40] # Boru karakterini temizle, çok uzunsa kes
            e_type = el.get("type", "unk")[:10]
            center = el.get("center", {})
            x, y = center.get("x", 0), center.get("y", 0)
            
            # Format: 1 | Save | button | (100,200)
            lines.append(f"{e_id} | {name} | {e_type} | ({x},{y})")
            
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
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload)}
        ]
        
        return messages, llm_view
    
    def _parse_plan(self, content: str) -> Dict:
        # DEBUG LOG: Raw LLM response
        logger.info("=" * 80)
        logger.info("📥 RECEIVED FROM LLM")
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
            logger.error("❌ JSON PARSE ERROR: %s", str(e))
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
        # 🚫 CRITICAL: Disable the "Smart" features
        self._use_semantic_filter = False 
        self.max_attempts = 1
    
    async def _fetch_state(self, attempt: int = 1) -> Dict:
        """Override to ALWAYS use ODS URL, ignoring attempt number/logic."""
        # Force ODS URL
        url = self.state_url_ods
        logger.info("📸 Fetching RAW State from ODS: %s", url)
        
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
        
        logger.warning("⚠️ RUNNING IN RAW ODS MODE (Benchmark Mode) ⚠️")
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
            logger.info("✓ RAW MODE SUCCESS (Surprisingly!)")
            return RunResult("passed", 1, actions_log, final_state, plan, "Success (Raw ODS)")
        
        logger.info("✗ RAW MODE FAILED (As expected?)")
        return RunResult("failed", 1, actions_log, final_state, plan, "Validation Failed")
