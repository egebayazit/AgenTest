# llm_backend.py v2.1
# Complete rewrite with modular validation system + semantic filtering

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

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a Windows UI automation expert. Execute test steps using ONLY elements from current_state.

OUTPUT: Valid JSON only. No markdown, no explanations.

TASK: Parse the test step, find the correct element, and click it.

CURRENT_STATE FORMAT:
ID | Name | Type | (x,y)
------------------------------------------------------------
1 | Save Settings | text | (10,20)
2 | [Icon] | icon | (100,20)

PARSING TEST STEP:
Extract these components:
1. CONTROL TYPE: What to click (checkbox, button, icon, toggle, dropdown)
2. ANCHOR NAME: Reference element name (often after "next to", "in", "of")

SEARCH STRATEGY:

STEP 1: Find Anchor Element
- Search Name column for ANCHOR text
- If found: Note ID, Type, and (x,y)
- If not found: Return empty steps

STEP 2: Determine if Neighbor Search Needed
Check if CONTROL TYPE is different from ANCHOR Type:

A. CONTROL TYPE mentions: checkbox, toggle, switch, icon, dropdown, arrow
   AND
   ANCHOR Type = text OR Name doesn't match CONTROL TYPE
   → NEIGHBOR SEARCH REQUIRED

B. CONTROL TYPE matches ANCHOR Name directly
   → CLICK ANCHOR DIRECTLY

STEP 3: Neighbor Search (if required)
Scan elements with:
- Same Y coordinate (±15 pixels)
- Within 300 pixels on X axis
- Type = icon (most controls are icons)
- ID different from Anchor ID

Neighbor Priority (check in order):
1. Name contains CONTROL TYPE keyword:
   - checkbox → "check", "tick", "mark", "box"
   - toggle → "toggle", "switch", "on", "off"
   - dropdown → "drop", "arrow", "down", "menu"
   - icon → any icon type
   
2. Closest to anchor (prefer <200px distance)

3. If multiple matches: Pick closest on X axis

STEP 4: Final Click Target
- If neighbor found → Click neighbor coordinates
- If no neighbor → Click anchor (fallback)

EXAMPLES:

Example 1: Checkbox with Text Anchor
─────────────────────────────────────
Input:
  test_step: "Click 'checkbox' next to Main Rule Control"
  current_state:
    13 | Main Rule Control | text | (76,368)
    14 | checkmark | icon | (283,368)
    15 | Other Element | text | (76,400)

Analysis:
  1. CONTROL TYPE = "checkbox", ANCHOR = "Main Rule Control"
  2. Find anchor: ID 13 found
  3. Anchor Type = text ≠ checkbox → Neighbor search needed
  4. Search Y=368±15 (353 to 383):
     - ID 14: Y=368✓, Type=icon✓, Name has "check"✓, Distance=207px✓
  5. Neighbor found! Click ID 14

Output:
{
  "action_id": "step_1",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":1,"target":{"point":{"x":283,"y":368}}}],
  "reasoning": "Anchor 'Main Rule Control' at ID 13 (76,368). Control type 'checkbox' needs neighbor. Found checkmark icon at ID 14 (283,368) on same row. Clicking neighbor."
}

Example 2: Direct Button Click
───────────────────────────────
Input:
  test_step: "Click 'Load Scenario' button"
  current_state:
    26 | Load Scenario | button | (1561,797)
    27 | Cancel | button | (1650,797)

Analysis:
  1. CONTROL TYPE = "button", ANCHOR = "Load Scenario"
  2. Find anchor: ID 26 found
  3. Anchor Type = button = CONTROL TYPE → Direct click
  4. No neighbor search needed

Output:
{
  "action_id": "step_1",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":1,"target":{"point":{"x":1561,"y":797}}}],
  "reasoning": "Found 'Load Scenario' button at ID 26 (1561,797). Direct match, no neighbor needed. Clicking anchor."
}

Example 3: Toggle with No Neighbor
───────────────────────────────────
Input:
  test_step: "Click 'toggle' next to Settings"
  current_state:
    5 | Settings | text | (50,100)
    6 | Profile | text | (50,130)

Analysis:
  1. CONTROL TYPE = "toggle", ANCHOR = "Settings"
  2. Find anchor: ID 5 found
  3. Anchor Type = text ≠ toggle → Neighbor search needed
  4. Search Y=100±15 (85 to 115):
     - ID 6: Y=130 (outside range)
     - No neighbors found
  5. Fallback: Click anchor

Output:
{
  "action_id": "step_1",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":1,"target":{"point":{"x":50,"y":100}}}],
  "reasoning": "Anchor 'Settings' at ID 5 (50,100). Control type 'toggle' needs neighbor but none found in same row. Fallback: clicking anchor."
}

Example 4: Dropdown Near Text
──────────────────────────────
Input:
  test_step: "Click dropdown next to Country"
  current_state:
    8 | Country | text | (200,300)
    9 | arrow down | icon | (350,302)
    10 | [Separator] | icon | (360,302)

Analysis:
  1. CONTROL TYPE = "dropdown", ANCHOR = "Country"
  2. Find anchor: ID 8 found
  3. Anchor Type = text ≠ dropdown → Neighbor search needed
  4. Search Y=300±15 (285 to 315):
     - ID 9: Y=302✓, Type=icon✓, Name has "arrow"✓, Distance=150px✓
     - ID 10: Y=302✓, Type=icon✓, Name generic, Distance=160px
  5. ID 9 better match (has "arrow" keyword)

Output:
{
  "action_id": "step_1",
  "coords_space": "physical",
  "steps": [{"type":"click","button":"left","click_count":1,"target":{"point":{"x":350,"y":302}}}],
  "reasoning": "Anchor 'Country' at ID 8 (200,300). Control type 'dropdown' needs neighbor. Found 'arrow down' icon at ID 9 (350,302) on same row. Clicking neighbor."
}

Example 5: Element Not Found
─────────────────────────────
Input:
  test_step: "Click 'checkbox' next to Advanced Options"
  current_state:
    1 | Settings | text | (50,50)
    2 | Profile | text | (50,80)

Analysis:
  1. CONTROL TYPE = "checkbox", ANCHOR = "Advanced Options"
  2. Find anchor: NOT FOUND
  3. Return empty steps

Output:
{
  "action_id": "step_1",
  "coords_space": "physical",
  "steps": [],
  "reasoning": "Anchor 'Advanced Options' not found in current_state. Cannot proceed."
}

CONTROL TYPE KEYWORDS (for neighbor matching):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
checkbox  → check, tick, mark, box, select
toggle    → toggle, switch, on, off, enable
dropdown  → drop, arrow, down, expand, menu, combo
icon      → (any icon type element)
button    → (usually direct match, no neighbor needed)

REASONING FORMAT (mandatory):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Must include:
1. Anchor element: Name, ID, coordinates
2. Control type requested
3. Neighbor search result (if applicable)
4. Final decision: Which ID clicking and why

CRITICAL RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Neighbor search range: Y ±15px, X ≤300px
2. Neighbor must be Type=icon in most cases
3. If neighbor found, MUST click neighbor not anchor
4. If no neighbor found, fallback to anchor
5. Always explain neighbor search in reasoning
6. Empty steps only if anchor not found
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
            "maxItems": 3,
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
# VALIDATION SYSTEM - MODULAR DESIGN
# ============================================================================

class ExpectedResultValidator:
    """
    Modular validation system for expected results
    
    Design:
    1. Parse expected result into atomic conditions
    2. Validate each condition independently
    3. ALL conditions must pass (AND logic)
    """
    
    ACTION_VERBS = {
        'closing': ['closes', 'closed', 'disappears', 'disappeared', 'hides', 'hidden', 'gone'],
        'loading': ['loads', 'loaded', 'loading', 'opens', 'opened', 'opening', 'appears', 'appeared'],
        'visibility': ['visible', 'shown', 'displayed', 'shows', 'displays'],
        'selection': ['selected', 'highlighted', 'checked', 'active', 'focused'],
        'state_change': ['starts', 'started', 'stops', 'stopped', 'runs', 'running', 'pauses', 'paused']
    }
    
    # Context-aware keywords: boost matching for element types
    ELEMENT_TYPE_KEYWORDS = {
        'dialog': ['dialog', 'dialogue', 'window', 'modal', 'popup'],
        'panel': ['panel', 'pane', 'sidebar', 'toolbar'],
        'button': ['button', 'btn'],
        'list': ['list', 'listbox', 'listview', 'grid'],
        'field': ['field', 'textbox', 'input', 'edit'],
        'label': ['label', 'text', 'caption'],
    }
    
    def validate(
        self,
        expected_result: str,
        visible_names: List[str],
        ui_changed: bool,
        change_magnitude: float
    ) -> Tuple[bool, str]:
        """
        Main validation entry point
        
        Returns:
            (passed, reason)
        """
        
        # Parse into atomic conditions
        conditions = self._parse_conditions(expected_result)
        
        logger.info("📋 Parsed %d condition(s):", len(conditions))
        for i, cond in enumerate(conditions, 1):
            logger.info("  %d. Type=%s, Subject='%s', Action='%s'", 
                       i, cond['type'], cond.get('subject', ''), cond.get('action', ''))
        
        # Validate each condition
        for i, cond in enumerate(conditions, 1):
            passed, reason = self._validate_single_condition(
                cond, visible_names, ui_changed, change_magnitude
            )
            
            if passed:
                logger.info("  ✓ Condition %d PASS: %s", i, reason)
            else:
                logger.info("  ✗ Condition %d FAIL: %s", i, reason)
                return (False, f"Condition {i} failed: {reason}")
        
        logger.info("✓ Expected HOLDS: All %d conditions passed", len(conditions))
        return (True, f"All {len(conditions)} conditions passed")
    
    def _parse_conditions(self, expected_result: str) -> List[Dict[str, Any]]:
        """Parse expected result into atomic conditions"""
        
        # Check for combined conditions (X and Y)
        if " and " in expected_result.lower():
            # PRIORITY 1: Check quoted + visibility pattern FIRST
            # Example: "'F-35_1' is highlighted and player info panel visible"
            parts = expected_result.split(" and ", 1)
            quoted_left = re.findall(r'["\']([^"\']+)["\']', parts[0])
            
            visibility_keywords = ['visible', 'panel', 'shown', 'displayed', 'appears', 'open']
            right_has_visibility = any(kw in parts[1].lower() for kw in visibility_keywords)
            
            if quoted_left and right_has_visibility:
                logger.debug("  → Pattern: quoted + visibility")
                return [
                    self._parse_single(parts[0].strip()),
                    self._parse_single(parts[1].strip())
                ]
            
            # PRIORITY 2: Check quoted + quoted
            quoted_right = re.findall(r'["\']([^"\']+)["\']', parts[1])
            if quoted_left and quoted_right:
                logger.debug("  → Pattern: quoted + quoted")
                return [
                    self._parse_single(parts[0].strip()),
                    self._parse_single(parts[1].strip())
                ]
            
            # PRIORITY 3: Try same-subject multi-action
            # Example: "dialog closes and loads"
            same_subject = self._parse_same_subject_multi_action(expected_result)
            if same_subject:
                return same_subject
        
        # Single condition
        return [self._parse_single(expected_result)]
    
    def _parse_same_subject_multi_action(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse: "<subject> <action1> and <action2>"
        Example: "dialog closes and loads"
        """
        
        # Find all action verbs
        found_actions = []
        text_lower = text.lower()
        
        for action_type, verbs in self.ACTION_VERBS.items():
            for verb in verbs:
                if verb in text_lower:
                    pos = text_lower.find(verb)
                    found_actions.append((action_type, verb, pos))
        
        if len(found_actions) < 2:
            return None
        
        # Sort by position in text
        found_actions.sort(key=lambda x: x[2])
        
        # Check if "and" is between first two actions
        first_verb_pos = found_actions[0][2]
        second_verb_pos = found_actions[1][2]
        
        between_text = text_lower[first_verb_pos:second_verb_pos]
        if " and " not in between_text:
            return None
        
        # Extract subject (before first action)
        subject = text[:first_verb_pos].strip()
        
        # Subject should be meaningful
        if len(subject) < 3 or subject.lower() in ['the', 'a', 'an', 'it']:
            return None
        
        # Create conditions for first two actions
        conditions = []
        for action_type, verb, _ in found_actions[:2]:
            cond = {
                'type': action_type,
                'subject': subject,
                'action': verb,
                'targets': {subject.lower()},
                'requires_ui_change': (action_type in ['loading', 'visibility', 'state_change'])
            }
            conditions.append(cond)
        
        logger.debug("  → Detected same-subject multi-action: %s (%s + %s)",
                    subject, found_actions[0][1], found_actions[1][1])
        
        return conditions
    
    def _parse_single(self, text: str) -> Dict[str, Any]:
        """Parse a single condition"""
        
        text_lower = text.lower()
        
        # FIRST: Strip auxiliary verbs and articles BEFORE any processing
        # This prevents "is visible" from capturing "is" as a target word
        aux_verbs = r'\b(is|are|was|were|be|been|being|am|the|a|an)\b'
        text_cleaned = re.sub(aux_verbs, ' ', text_lower).strip()
        text_cleaned = re.sub(r'\s+', ' ', text_cleaned)  # Collapse multiple spaces
        
        # Detect condition type (use cleaned text for better detection)
        condition_type = 'generic'
        action_verb = None
        
        for ctype, verbs in self.ACTION_VERBS.items():
            for verb in verbs:
                if verb in text_cleaned:
                    condition_type = ctype
                    action_verb = verb
                    break
            if condition_type != 'generic':
                break
        
        # Extract quoted items (use original text to preserve exact names)
        quoted_items = re.findall(r'["\']([^"\']+)["\']', text)
        
        if quoted_items:
            subject = quoted_items[0]
            targets = {item.lower() for item in quoted_items}
        elif condition_type != 'generic' and action_verb:
            pos = text_cleaned.find(action_verb)
            subject = text_cleaned[:pos].strip() if pos > 0 else None
            targets = {subject} if subject else set()
        else:
            # Extract words from CLEANED text (auxiliary verbs already removed)
            words = re.findall(r'\b\w{3,}\b', text_cleaned)
            stopwords = {'and', 'or', 'with', 'that', 'this', 'from', 'for', 'row'}
            targets = set(words) - stopwords
            subject = None
        
        return {
            'type': condition_type,
            'subject': subject,
            'action': action_verb,
            'targets': targets,
            'requires_ui_change': (condition_type in ['loading', 'visibility', 'state_change'])
        }
    
    def _match_targets_in_ui(
        self,
        targets: Set[str],
        visible_names: List[str],
        match_mode: str = 'partial'  # 'exact', 'partial', 'word_based'
    ) -> Tuple[int, int, float]:
        """
        Universal matching helper for all condition types
        
        Args:
            targets: Set of target strings to find
            visible_names: List of visible UI element names (already lowercase)
            match_mode: 
                - 'exact': Full string must match (for closing validation)
                - 'partial': Substring match (for selection)
                - 'word_based': Split into words and match (for visibility, multi-word phrases)
        
        Returns:
            (found_count, total_targets, coverage)
        """
        
        if not targets:
            return 0, 0, 0.0
        
        def fuzzy_match(target: str, ui_element: str) -> bool:
            """
            Check if target matches ui_element with fuzzy tolerance
            
            Rules:
            - If target is substring of ui_element → TRUE (exact partial match)
            - If target >5 chars and Levenshtein distance ≤2 → TRUE (typo tolerance)
            
            Examples:
                "management" in "scenario management" → TRUE (substring)
                "managment" vs "management" → TRUE (distance=1, typo)
                "plyr" vs "player" → FALSE (distance=3, too different)
            """
            # First: try exact substring match (fast path)
            if target in ui_element:
                return True
            
            # Second: fuzzy match for longer words (typo tolerance)
            if len(target) > 5:
                # Use SequenceMatcher for similarity
                similarity = SequenceMatcher(None, target, ui_element).ratio()
                
                # If very similar (>0.85), accept
                if similarity > 0.85:
                    return True
                
                # Also check if target appears as fuzzy substring in ui_element
                # Split ui_element into words and check each
                for word in ui_element.split():
                    if len(word) > 5:
                        word_similarity = SequenceMatcher(None, target, word).ratio()
                        if word_similarity > 0.85:  # ~1-2 char difference for 6-8 char words
                            return True
            
            return False
        
        if match_mode == 'exact':
            # Exact match: target must appear as-is in UI
            found_count = sum(1 for t in targets if any(t == name for name in visible_names))
            return found_count, len(targets), found_count / len(targets)
        
        elif match_mode == 'partial':
            # Partial match with fuzzy tolerance
            found_count = sum(1 for t in targets 
                            if any(fuzzy_match(t, name) for name in visible_names))
            return found_count, len(targets), found_count / len(targets)
        
        elif match_mode == 'word_based':
            # Word-based: split multi-word targets into words, match individually
            all_target_words = set()
            
            for target in targets:
                if ' ' in target:  # Multi-word target
                    words = target.split()
                    all_target_words.update(words)
                    all_target_words.add(target)  # Also keep full phrase
                else:
                    all_target_words.add(target)
            
            # Remove stopwords
            stopwords = {'the', 'is', 'a', 'an', 'of', 'in', 'on', 'at', 'row', 'are', 'was', 'were'}
            all_target_words = all_target_words - stopwords
            
            if not all_target_words:
                return 0, 0, 0.0
            
            # Count matches with fuzzy tolerance
            found_count = sum(1 for word in all_target_words 
                            if any(fuzzy_match(word, name) for name in visible_names))
            
            return found_count, len(all_target_words), found_count / len(all_target_words)
        
        else:
            raise ValueError(f"Unknown match_mode: {match_mode}")
    
    def _validate_single_condition(
        self,
        condition: Dict[str, Any],
        visible_names: List[str],
        ui_changed: bool,
        change_magnitude: float
    ) -> Tuple[bool, str]:
        """Validate a single condition using universal matching"""
        
        cond_type = condition['type']
        targets = condition['targets']
        
        if cond_type == 'closing':
            # Elements must be ABSENT (use word-based to avoid false positives)
            # Example: "Scenario Management dialog" should NOT match just "Scenario" alone
            found, total, coverage = self._match_targets_in_ui(targets, visible_names, 'word_based')
            
            # For closing, if ANY significant portion found (>30%), consider NOT closed
            if coverage > 0.3:
                return (False, f"{found}/{total} words still present ({coverage:.0%})")
            return (True, "All targets absent")
        
        elif cond_type == 'loading':
            # MUST have UI change
            if not ui_changed or change_magnitude < 0.05:
                return (False, f"No UI change ({change_magnitude:.1f}%)")
            return (True, f"UI changed {change_magnitude:.1f}%")
        
        elif cond_type == 'visibility':
            # Elements must be PRESENT (word-based for multi-word phrases)
            found, total, coverage = self._match_targets_in_ui(targets, visible_names, 'word_based')
            
            if total == 0:
                return (False, "No valid targets")
            
            # Accept if >=50% of words found
            if coverage >= 0.5:
                return (True, f"{found}/{total} target words visible ({coverage:.0%})")
            return (False, f"Only {found}/{total} found ({coverage:.0%})")
        
        elif cond_type == 'selection':
            # Elements must exist (partial match - substring is OK)
            found, total, coverage = self._match_targets_in_ui(targets, visible_names, 'partial')
            
            if found > 0:
                return (True, f"Target found (UI changed={ui_changed})")
            return (False, "Target not found")
        
        elif cond_type == 'state_change':
            # State keyword must exist (partial match)
            action = condition.get('action', '')
            if not action:
                return (False, "No action keyword")
            
            # Check if action keyword appears in UI (case-insensitive substring)
            if any(action in name for name in visible_names):
                return (True, f"State '{action}' found")
            
            # Also check for related forms (e.g., "running" vs "run")
            # Strip common suffixes
            action_stem = action.rstrip('ing').rstrip('ed').rstrip('s')
            if len(action_stem) >= 3 and any(action_stem in name for name in visible_names):
                return (True, f"State '{action_stem}' found")
            
            return (False, f"State '{action}' not found")
        
        else:  # generic fuzzy match
            # Use word-based for generic matching
            found, total, coverage = self._match_targets_in_ui(targets, visible_names, 'word_based')
            
            if total == 0:
                return (False, "No valid targets")
            
            if coverage >= 0.4:
                return (True, f"{coverage:.0%} match ({found}/{total})")
            return (False, f"Only {coverage:.0%} match")

# ============================================================================
# LLM BACKEND
# ============================================================================

class LLMBackend:
    """LLM-based action planner with modular validation"""
    
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
        max_tokens: int = 384,
        max_plan_steps: int = 10,
        schema_retry_limit: int = 1,
        http_referrer: str = "https://agentest.local/backend",
        client_title: str = "AgenTest LLM Backend",
        enforce_json_response: bool = True,
    ) -> None:
        
        # Validation
        if not llm_model:
            raise ValueError("llm_model is required")
        if llm_provider not in ("ollama", "openrouter", "lmstudio"):
            raise ValueError("llm_provider must be 'ollama', 'lmstudio' or 'openrouter'")
        if llm_provider == "openrouter" and not llm_api_key:
            raise ValueError("llm_api_key required for OpenRouter")
        
        # Store config
        self.state_url_windriver = state_url_windriver
        self.state_url_ods = state_url_ods
        self.max_attempts = max_attempts
        self.llm_provider = llm_provider
        self.model = llm_model
        self.llm_base_url = llm_base_url.rstrip("/")
        self.api_key = llm_api_key
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
        
        # Validation system
        self.validator = ExpectedResultValidator()
        
        # Semantic filtering system (NEW in v2.1)
        self.semantic_filter = SemanticStateFilter(
            row_tolerance=15,      # Same-row detection tolerance (pixels)
            match_threshold=0.70   # Fuzzy match threshold (70% similarity)
        )
        self._use_semantic_filter = True  # Enable/disable filtering
    
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
    # COORDINATE-BASED VALIDATION HELPERS
    # ========================================================================
    
    def _check_coordinate_element_change(
        self,
        clicked_coords: Optional[Tuple[int, int]],
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        tolerance: int = 5
    ) -> Tuple[bool, str]:
        """
        Check if element at clicked coordinates changed (name/presence)
        
        Generic validation for all control types:
        - Checkbox: "checkmark" → "maximize window" (ODS naming quirk)
        - Toggle: "toggle off" → "toggle on"
        - Dropdown: "arrow down" → "arrow up"
        - Radio: "empty circle" → "filled circle"
        - Any control: element disappears or name changes
        
        Args:
            clicked_coords: (x, y) coordinates that were clicked
            state_before: UI state before click
            state_after: UI state after click
            tolerance: Pixel tolerance for coordinate matching (default 5px)
        
        Returns:
            (changed, description) - True if element changed at coordinates
        """
        if not clicked_coords:
            return (False, "No clicked coordinates provided")
        
        click_x, click_y = clicked_coords
        
        # Find element at clicked coords in BEFORE state
        elem_before = None
        for elem in state_before.get("elements", []):
            center = elem.get("center", {})
            ex, ey = center.get("x", -999), center.get("y", -999)
            
            if abs(ex - click_x) <= tolerance and abs(ey - click_y) <= tolerance:
                elem_before = elem
                break
        
        if not elem_before:
            logger.debug("  → Coordinate validation: No element at (%d,%d) in BEFORE state", 
                        click_x, click_y)
            return (False, "No element found at clicked coordinates (before)")
        
        name_before = elem_before.get("name", "").strip().lower()
        type_before = elem_before.get("type", "").strip().lower()
        
        # Find element at same coords in AFTER state
        elem_after = None
        for elem in state_after.get("elements", []):
            center = elem.get("center", {})
            ex, ey = center.get("x", -999), center.get("y", -999)
            
            if abs(ex - click_x) <= tolerance and abs(ey - click_y) <= tolerance:
                elem_after = elem
                break
        
        # CASE 1: Element disappeared
        if not elem_after:
            logger.info("  ✓ Coordinate validation: Element disappeared at (%d,%d)", 
                       click_x, click_y)
            logger.info("     Before: '%s' (type=%s)", name_before[:50], type_before)
            logger.info("     After:  (element gone)")
            return (True, f"Element at ({click_x},{click_y}) disappeared")
        
        name_after = elem_after.get("name", "").strip().lower()
        type_after = elem_after.get("type", "").strip().lower()
        
        # CASE 2: Name changed
        if name_before != name_after:
            logger.info("  ✓ Coordinate validation: Element name changed at (%d,%d)", 
                       click_x, click_y)
            logger.info("     Before: '%s' (type=%s)", name_before[:50], type_before)
            logger.info("     After:  '%s' (type=%s)", name_after[:50], type_after)
            return (True, f"Element name changed at ({click_x},{click_y})")
        
        # CASE 3: Type changed (rare but possible)
        if type_before != type_after:
            logger.info("  ✓ Coordinate validation: Element type changed at (%d,%d)", 
                       click_x, click_y)
            logger.info("     Before: type=%s", type_before)
            logger.info("     After:  type=%s", type_after)
            return (True, f"Element type changed at ({click_x},{click_y})")
        
        # CASE 4: No change
        logger.debug("  → Coordinate validation: No change at (%d,%d)", click_x, click_y)
        logger.debug("     Name: '%s'", name_before[:50])
        return (False, f"No change detected at clicked coordinates ({click_x},{click_y})")
    
    # SCENARIO & STEP EXECUTION
    # ========================================================================
    
    async def run_scenario(
        self,
        steps: List[StepDefinition],
        *,
        temperature: float = 0.1,
    ) -> ScenarioResult:
        """Run a full test scenario"""
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
        """Run a single test step"""
        history_payload: List[Dict[str, Any]] = list(recent_actions or [])
        actions_log: List[ActionExecutionLog] = []
        last_plan: Optional[Dict[str, Any]] = None
        state: Dict[str, Any] = {}
        
        for attempt in range(1, self.max_attempts + 1):
            detection_method = "WinDriver" if attempt == 1 else "ODS"
            logger.info("Attempt %d/%d (%s)", attempt, self.max_attempts, detection_method)
            
            # Fetch state
            state = await self._fetch_state(attempt_number=attempt)
            
            # Request plan
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
                
                # Validate plan
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
                        message = "LLM plan missing valid 'steps'"
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
                logger.error("❌ LLM generated %d steps (max: %d)", len(steps), self.max_plan_steps)
                plan["steps"] = []
                plan["reasoning"] = f"Rejected: {len(steps)} steps exceeds limit"
                plan["_backend_steps_substituted"] = True
                steps = []
                reasoning = plan["reasoning"]
            
            # Log plan
            logger.info("=" * 80)
            logger.info("📋 TEST STEP: %s", test_step)
            logger.info("🎯 EXPECTED: %s", expected_result[:100])
            logger.info("🤖 LLM PLAN: %d step(s)", len(steps))
            logger.info("💭 REASONING: %s", reasoning[:150])
            
            if steps:
                logger.info("📍 ACTIONS:")
                for i, step in enumerate(steps, 1):
                    step_type = step.get("type", "unknown")
                    if step_type == "click":
                        point = step.get("target", {}).get("point", {})
                        logger.info("  %d. CLICK at (%s, %s)", i, point.get('x'), point.get('y'))
                    elif step_type == "type":
                        logger.info("  %d. TYPE '%s...'", i, step.get("text", "")[:30])
                    else:
                        logger.info("  %d. %s", i, step_type.upper())
            else:
                logger.warning("  ⚠️  NO ACTIONS")
                if attempt < self.max_attempts:
                    logger.info("  → Retrying with ODS...")
                    continue
                else:
                    return RunResult(
                        status="failed",
                        attempts=attempt,
                        actions=actions_log,
                        final_state=state,
                        last_plan=plan,
                        reason=f"Element not found after {self.max_attempts} attempts",
                    )
            
            logger.info("=" * 80)
            
            # Execute action
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
                message = f"SUT error: {ack.get('message', '')}"
                logger.error("✗ %s", message)
                return RunResult(
                    status="error",
                    attempts=attempt,
                    actions=actions_log,
                    final_state=final_state,
                    last_plan=plan,
                    reason=message,
                )
            
            # Fetch state after
            final_state = await self._fetch_state_safe(state, attempt_number=attempt)
            log_entry.state_after = final_state
            
            # Detect UI change
            ui_changed, change_magnitude = self._detect_ui_change(state_before, final_state)
            
            logger.info("🔍 VALIDATING...")
            logger.info("  Expected: %s", expected_result[:100])
            logger.info("  UI changed: %s (%.1f%%)", "YES" if ui_changed else "NO", change_magnitude * 100)
            
            # Validate expected result
            validation_passed = self._expected_holds(
                final_state,
                expected_result,
                plan,
                state_before=state_before
            )
            
            if validation_passed:
                logger.info("✓ SUCCESS")
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
                
                if attempt >= self.max_attempts:
                    return RunResult(
                        status="failed",
                        attempts=attempt,
                        actions=actions_log,
                        final_state=final_state,
                        last_plan=plan,
                        reason=f"Expected result not achieved after {attempt} attempts",
                    )
                
                logger.info("  → Retrying with ODS...")
        
        return RunResult(
            status="failed",
            attempts=self.max_attempts,
            actions=actions_log,
            final_state=final_state,
            last_plan=last_plan,
            reason=f"Exhausted {self.max_attempts} attempts",
        )
    
    # ========================================================================
    # STATE & ACTION COMMUNICATION
    # ========================================================================
    
    async def _fetch_state(self, attempt_number: int = 1) -> Dict[str, Any]:
        """Fetch current state from SUT"""
        use_ods = (attempt_number == 2)
        state_url = self.state_url_ods if use_ods else self.state_url_windriver
        source_name = "ODS" if use_ods else "WinDriver"
        
        logger.info("Fetching state (%s): %s", source_name, state_url)
        
        try:
            async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
                response = await client.post(state_url, json={})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            message = f"Failed to contact SUT: {exc}"
            logger.error(message)
            raise SUTCommunicationError(message) from exc
        
        try:
            state = response.json()
        except json.JSONDecodeError as exc:
            message = "SUT returned invalid JSON"
            logger.error(message)
            raise SUTCommunicationError(message) from exc
        
        element_count = len(state.get("elements", []))
        logger.debug("  → Received %d elements", element_count)
        
        return state
    
    async def _fetch_state_safe(self, fallback: Dict[str, Any], attempt_number: int = 1) -> Dict[str, Any]:
        """Fetch state with fallback"""
        try:
            return await self._fetch_state(attempt_number)
        except BackendError:
            logger.warning("Failed to fetch state, using fallback")
            return fallback
    
    async def _send_action(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Send action plan to SUT"""
        try:
            async with httpx.AsyncClient(timeout=self.sut_timeout) as client:
                response = await client.post(self.action_url, json=plan)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            message = f"Failed to send action: {exc}"
            logger.error(message)
            raise SUTCommunicationError(message) from exc
        
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            message = "SUT returned invalid JSON"
            logger.error(message)
            raise SUTCommunicationError(message) from exc
    
    # ========================================================================
    # UI CHANGE DETECTION
    # ========================================================================
    
    def _compute_state_hash(self, state: Dict[str, Any]) -> str:
        """Compute hash of UI state"""
        elements = state.get("elements", [])
        
        state_signature = []
        for elem in elements:
            name = elem.get("name", "")
            center = elem.get("center", {})
            
            # Include selection state
            is_selected = elem.get("is_selected", False)
            selection_state = elem.get("selection_state", "")
            
            sig = f"{name}@{center.get('x', 0)},{center.get('y', 0)}"
            if is_selected or selection_state:
                sig += f"|sel:{is_selected}:{selection_state}"
            
            state_signature.append(sig)
        
        signature_str = "|".join(sorted(state_signature))
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def _detect_ui_change(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Detect UI change"""
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
    # EXPECTED RESULT VALIDATION
    # ========================================================================
    
    def _expected_holds(
        self,
        state: Dict[str, Any],
        expected_result: str,
        plan_executed: Dict[str, Any],
        state_before: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validate expected result using modular validation system with coordinate-based checking"""
        
        if not expected_result:
            return False
        
        steps_executed = plan_executed.get("steps", [])
        if not steps_executed:
            logger.warning("❌ No actions executed")
            return False
        
        # Check if meaningful action executed
        meaningful_types = {"click", "type", "key_combo", "drag"}
        has_meaningful = any(s.get("type") in meaningful_types for s in steps_executed)
        if not has_meaningful:
            logger.warning("❌ Only passive actions")
            return False
        
        # Extract clicked coordinates from first click action
        clicked_coords = None
        for step in steps_executed:
            if step.get("type") == "click":
                point = step.get("target", {}).get("point", {})
                if point and "x" in point and "y" in point:
                    clicked_coords = (int(point["x"]), int(point["y"]))
                    logger.debug("  → Clicked coordinates: %s", clicked_coords)
                    break
        
        # Get visible elements
        elems = state.get("elements", [])
        if not elems:
            return False
        
        visible_names = [e.get("name", "").strip().lower() for e in elems if e.get("name")]
        
        # Calculate UI change
        ui_changed = False
        change_magnitude = 0.0
        if state_before is not None:
            ui_changed, change_magnitude = self._detect_ui_change(state_before, state)
        
        # COORDINATE-BASED VALIDATION (Generic for all controls)
        # Check if element at clicked coordinates changed
        coord_changed = False
        coord_reason = ""
        if clicked_coords and state_before:
            coord_changed, coord_reason = self._check_coordinate_element_change(
                clicked_coords,
                state_before,
                state,
                tolerance=5  # 5 pixel tolerance
            )
            
            if coord_changed:
                logger.info("✓ COORDINATE VALIDATION PASSED: %s", coord_reason)
                # For control interactions (checkbox, toggle, dropdown, etc.),
                # coordinate change is sufficient proof
                return True
        
        # Fallback to standard validation if coordinate check didn't pass
        passed, reason = self.validator.validate(
            expected_result,
            visible_names,
            ui_changed,
            change_magnitude
        )
        
        if passed:
            logger.info("✓ Expected HOLDS: %s", reason)
        else:
            logger.info("✗ Expected NOT met: %s", reason)
            # Log additional context about coordinate check
            if clicked_coords:
                logger.info("  → Coordinate check: %s", coord_reason)
        
        return passed
    
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
        """Request action plan from LLM"""
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
        """Request from Ollama"""
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
        
        try:
            async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                response = await client.post(url, json=body)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMCommunicationError(f"Ollama request failed: {exc}") from exc
        
        try:
            data = response.json()
            content = data["message"]["content"].strip()
        except (KeyError, json.JSONDecodeError) as exc:
            raise LLMCommunicationError(f"Invalid Ollama response: {exc}") from exc
        
        if not content:
            raise PlanParseError("Ollama returned empty content")
        
        return self._parse_plan(content)
    
    async def _request_plan_lmstudio(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        """Request from LM Studio"""
        url = f"{self.llm_base_url}/chat/completions"
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        
        if self.enforce_json_response:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "AgenTestPlan",
                    "schema": AGEN_TEST_PLAN_SCHEMA,
                },
            }
        
        try:
            async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMCommunicationError(f"LM Studio failed: {exc}") from exc
        
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMCommunicationError(f"Invalid LM Studio response: {exc}") from exc
        
        if not content:
            raise PlanParseError("LM Studio returned empty content")
        
        return self._parse_plan(content)
    
    async def _request_plan_openrouter(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        """Request from OpenRouter"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.http_referrer,
            "X-Title": self.client_title,
            "Content-Type": "application/json",
        }
        
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        
        if self.enforce_json_response:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "AgenTestPlan",
                    "schema": AGEN_TEST_PLAN_SCHEMA,
                },
            }
        
        try:
            async with httpx.AsyncClient(timeout=self.llm_timeout) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMCommunicationError(f"OpenRouter failed: {exc}") from exc
        
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMCommunicationError(f"Invalid OpenRouter response: {exc}") from exc
        
        if not content:
            raise PlanParseError("OpenRouter returned empty content")
        
        return self._parse_plan(content)
    
    def _build_messages(
        self,
        *,
        test_step: str,
        expected_result: str,
        note_to_llm: Optional[str],
        state: Dict[str, Any],
        recent_actions: List[Dict[str, Any]],
        schema_hint: Optional[str],
        attempt_number: int,
    ) -> List[Dict[str, Any]]:
        """Build messages for LLM with optional semantic filtering"""
        screen_info = state.get("screen", {})
        
        # SEMANTIC FILTERING (NEW in v2.1)
        if self._use_semantic_filter and "elements" in state:
            logger.info("🔍 Applying semantic filtering...")
            
            original_elements = state["elements"]
            original_count = len(original_elements)
            
            # Filter elements based on test step, expected result, and note
            filtered_elements = self.semantic_filter.filter_elements(
                all_elements=original_elements,
                test_step=test_step,
                expected_result=expected_result,
                note_to_llm=note_to_llm or ""
            )
            
            filtered_count = len(filtered_elements)
            reduction_pct = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0
            
            logger.info("  → Elements: %d → %d (%.1f%% reduction)", 
                       original_count, filtered_count, reduction_pct)
            
            # Build filtered llm_view
            llm_view, id_map = self.semantic_filter.format_for_llm(filtered_elements)
            
            # Log filtered llm_view
            logger.info("📋 Filtered LLM View (%d chars, %d lines):", len(llm_view), llm_view.count('\n') + 1)
            logger.info("─" * 80)
            for line in llm_view.split('\n'):
                logger.info("  %s", line)
            logger.info("─" * 80)
            
            # Store id_map for later reference (if needed for action execution)
            state["_filtered_id_map"] = id_map
            
        else:
            # Use original llm_view from state
            llm_view = state.get("llm_view", "")
            if not llm_view:
                logger.warning("⚠️  No llm_view in state and filtering disabled!")
        
        payload: Dict[str, Any] = {
            "test_step": test_step,
            "expected_result": expected_result,
            "screen": {"w": screen_info.get("w"), "h": screen_info.get("h")},
            "current_state": llm_view,
        }
        
        if recent_actions:
            payload["recent_actions"] = recent_actions[-2:]
        
        if schema_hint:
            payload["backend_guidance"] = schema_hint
        
        if note_to_llm:
            payload["note_to_llm"] = note_to_llm
        
        if attempt_number > 1:
            payload["retry_context"] = {
                "attempt": attempt_number,
                "detection_method": "ODS",
                "message": f"Retry attempt {attempt_number} using ODS detection",
            }
        else:
            payload["retry_context"] = {
                "attempt": 1,
                "detection_method": "WinDriver",
                "message": "First attempt using WinDriver detection",
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
        """Parse LLM response"""
        stripped = self._strip_wrappers(content).strip()
        if not stripped:
            raise PlanParseError("Empty plan")
        
        try:
            plan = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Invalid JSON: {exc}") from exc
        
        if not isinstance(plan, dict):
            raise PlanParseError("Plan must be dict")
        
        if "steps" not in plan or not isinstance(plan.get("steps"), list):
            plan["steps"] = []
            plan["_backend_steps_substituted"] = True
        
        if "action_id" not in plan:
            plan["action_id"] = f"step_{int(time.time() * 1000)}"
        if "coords_space" not in plan:
            plan["coords_space"] = "physical"
        
        return plan
    
    def _strip_wrappers(self, text: str) -> str:
        """Strip markdown and prefixes"""
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
        return text
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def _validate_plan_against_screen(self, plan: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
        """Validate plan coordinates"""
        screen = state.get("screen") or {}
        sw, sh = screen.get("w"), screen.get("h")
        if not isinstance(sw, int) or not isinstance(sh, int) or sw <= 0 or sh <= 0:
            return None
        
        def in_bounds(x: float, y: float) -> bool:
            return 0 <= x < sw and 0 <= y < sh
        
        for idx, step in enumerate(plan.get("steps", [])):
            t = step.get("type")
            if t == "click":
                point = step.get("target", {}).get("point")
                if not point:
                    return f"step[{idx}] CLICK missing point"
                px, py = point.get("x"), point.get("y")
                if not in_bounds(px, py):
                    return f"step[{idx}] CLICK ({px},{py}) outside screen"
        
        return None
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _summarise_for_prompt(self, entry: ActionExecutionLog) -> Dict[str, Any]:
        """Summarize action for prompt"""
        return {
            "action_id": entry.action_id,
            "steps_count": len(entry.plan.get("steps", [])),
            "ack_status": entry.ack.get("status"),
            "timestamp": entry.timestamp,
        }

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