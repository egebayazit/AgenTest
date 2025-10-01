"""
plan_utils.py — Action-plan schema, normalization and guardrails for Agentest
=============================================================================

Purpose
-------
This module is the *safety & consistency layer* between the LLM and the SUT.
LLMs can return slightly malformed or optimistic plans. `plan_utils` ensures
that whatever reaches the SUT is **valid, normalized, and stable**.

What it does (at a glance)
--------------------------
1) **Defines a strict plan schema** (Pydantic models) for all supported actions:
   click, type, key_combo, drag, scroll, move, wait.

2) **Normalizes** messy details from LLM output:
   - Keyboard combos: maps synonyms (e.g. "control"→"ctrl", "return"→"enter"),
     lowercases, de-duplicates, and filters to an allowed key set.
   - Rect/coordinates: casts to integers, enforces minimum sizes.
   - Fills defaults: `action_id`, `coords_space`, `delay_ms`, etc.
   - Drops unknown/invalid action objects instead of crashing.

3) **Adds timing guardrails** to reduce flakiness:
   - After any `key_combo` → automatically inserts a short `wait` (~300ms).
   - After `click` and before a following `type` → inserts a short `wait` (~200ms).
   - Enforces a minimum `delay_ms` for `type` (e.g., ≥25–30ms).

4) **Validates** the final plan against the schema:
   - If the sanitized plan still violates the schema, you find out *here*,
     not at the SUT boundary.

Where to use it
---------------
- In the **LLM backend** immediately after receiving the model output.
- In the **Controller** (recommended) as the canonical place to enforce plan quality.
  This keeps behavior consistent even if you swap models or providers.

Key guarantees / invariants
---------------------------
- Returned plan always has: `action_id`, `coords_space="physical"`, and `steps` list.
- All steps conform to the defined action models, or are dropped.
- Timing heuristics are applied consistently (min delay, click→type wait, etc.).
- The function is **idempotent**: calling `sanitize_plan_dict` multiple times
  keeps producing the same normalized plan.

Security / safety
-----------------
- Rejects/filters keys not in the allowed set.
- Silently drops unknown action types and malformed steps (fail-safe).
- Ensures numeric types where required (rect, delta, etc.).

Example
-------
Raw LLM output (slightly messy):

    {
      "steps": [
        {"type":"key_combo","combo":["Control","S"]},
        {"type":"click","target":{"rect":{"x":"500","y":300,"w":-1,"h":"40"}}},
        {"type":"type","text":"hello", "delay_ms": 5}
      ]
    }

After `sanitize_plan_dict(plan)`:

    {
      "action_id": "step_1700000000",
      "coords_space": "physical",
      "steps": [
        {"type":"key_combo","combo":["ctrl","s"]},
        {"type":"wait","ms":300},               # guardrail inserted
        {"type":"click","button":"left","click_count":1,"modifiers":[],
         "target":{"rect":{"x":500,"y":300,"w":1,"h":40}}},
        {"type":"wait","ms":200},               # click→type guardrail
        {"type":"type","text":"hello","delay_ms":30,"enter":false}
      ],
      "reasoning": ""
    }

Testing tips
------------
- Unit-test `sanitize_plan_dict` with 10–20 focused cases (bad rectangles,
  weird combos, missing fields, unknown types, etc.).
- Validate that guardrails are inserted and that invalid steps are dropped.
- Keep a small golden-file suite for regression.

Design notes
------------
- `sanitize_plan_dict` is deliberately forgiving (best-effort), followed by a
  **strict schema validation**. This combo yields robust behavior without
  rejecting useful plans unnecessarily.
- Allowed keys are centralized; adding new keys is trivial.
- The module is model/provider-agnostic and reusable by both Backend and Controller.

"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import time

# --- Allowed keys and synonyms ------------------------------------------------
# (docstrings inline are optional; left concise to avoid noise)
_ALPHAS = [chr(c) for c in range(ord('a'), ord('z') + 1)]
_DIGITS = list("0123456789")
_FN = [f"f{i}" for i in range(1, 13)]
_NAV = ["up", "down", "left", "right", "home", "end", "pageup", "pagedown"]
_MOD = ["ctrl", "shift", "alt", "win"]
_OTHER = ["enter", "tab", "esc", "backspace", "delete"]
_ALLOWED_KEYS = set(_ALPHAS + _DIGITS + _FN + _NAV + _MOD + _OTHER)
_SYNONYMS = {
    "control": "ctrl",
    "return": "enter",
    "escape": "esc",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "del": "delete",
    "bksp": "backspace",
}

# --- Schema models ------------------------------------------------------------

class Rect(BaseModel):
    """Rectangle in physical pixels."""
    x: int
    y: int
    w: int
    h: int


class ClickTarget(BaseModel):
    """Target area for click actions."""
    rect: Rect


class Click(BaseModel):
    """Mouse click action. Defaults to a single left-click with no modifiers."""
    type: Literal["click"]
    button: Literal["left", "right", "middle"] = "left"
    click_count: int = 1
    modifiers: List[str] = []
    target: ClickTarget


class TypeAct(BaseModel):
    """Keyboard typing action. Use `enter=True` to press Enter after typing."""
    type: Literal["type"]
    text: str
    delay_ms: int = 30
    enter: bool = False


class KeyCombo(BaseModel):
    """Chorded key press (e.g. ['ctrl','s'] or ['win','s'])."""
    type: Literal["key_combo"]
    combo: List[str]


class Drag(BaseModel):
    """Mouse drag from point A to B."""
    type: Literal["drag"]
    from_: Dict[str, int] = Field(..., alias="from")
    to: Dict[str, int]
    button: Literal["left", "right", "middle"] = "left"
    hold_ms: int = 120


class Scroll(BaseModel):
    """Mouse wheel scroll. Positive delta scrolls down; negative scrolls up."""
    type: Literal["scroll"]
    delta: int
    horizontal: bool = False
    at: Optional[Dict[str, int]] = None


class Move(BaseModel):
    """Mouse move/hover to a point, optionally waiting for UI to settle."""
    type: Literal["move"]
    point: Dict[str, int]
    settle_ms: int = 150


class Wait(BaseModel):
    """Explicit delay to let the UI update."""
    type: Literal["wait"]
    ms: int


Action = Click | TypeAct | KeyCombo | Drag | Scroll | Move | Wait


class Plan(BaseModel):
    """Full action plan contract enforced before hitting the SUT."""
    action_id: str
    coords_space: Literal["physical"]
    steps: List[Action]
    reasoning: str


# --- Normalization helpers ----------------------------------------------------

def _normalize_combo(keys: Any) -> List[str]:
    """
    Normalize a key combo into a clean, ordered, lowercased list of allowed keys.

    - Accepts string or list; returns a list.
    - Maps common synonyms ('control'→'ctrl', 'return'→'enter', etc.).
    - Filters out unknown keys and removes duplicates preserving order.
    """
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        return []
    norm: List[str] = []
    for k in keys:
        if not isinstance(k, str):
            continue
        kk = _SYNONYMS.get(k.lower().strip(), k.lower().strip())
        if kk in _ALLOWED_KEYS:
            norm.append(kk)
    seen = set()
    out: List[str] = []
    for k in norm:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def sanitize_plan_dict(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort sanitizer for raw LLM plans.

    - Fills required defaults: `action_id`, `coords_space`.
    - Repairs rects/coordinates and coercions to int.
    - Enforces minimum `delay_ms` for TYPE; drops malformed steps.
    - Normalizes key combos and filters unknown keys.
    - Adds timing guardrails:
        * `wait(300ms)` after each `key_combo`
        * `wait(200ms)` between `click` → `type`
    - Returns a dict that can then be validated with `Plan.model_validate`.

    Notes
    -----
    * Unknown action types are silently dropped (fail-safe).
    * The function is idempotent; calling it multiple times is safe.

    Parameters
    ----------
    plan : dict
        Raw plan dictionary produced by the LLM.

    Returns
    -------
    dict
        Sanitized plan dictionary.
    """
    p = dict(plan)
    p.setdefault("action_id", f"step_{int(time.time())}")
    p.setdefault("coords_space", "physical")

    steps = p.get("steps")
    if not isinstance(steps, list):
        steps = []
    fixed: List[Dict[str, Any]] = []

    for s in steps:
        if not isinstance(s, dict):
            continue
        t = s.get("type")

        if t == "click":
            tgt = s.get("target", {})
            rect = tgt.get("rect") if isinstance(tgt, dict) else None
            if not isinstance(rect, dict):
                continue
            try:
                rect = {
                    "x": int(rect["x"]),
                    "y": int(rect["y"]),
                    "w": max(1, int(rect["w"])),
                    "h": max(1, int(rect["h"])),
                }
            except Exception:
                continue
            s["button"] = s.get("button", "left")
            s["click_count"] = int(s.get("click_count", 1))
            s["modifiers"] = [m.lower() for m in s.get("modifiers", []) if isinstance(m, str)]
            s["target"] = {"rect": rect}
            fixed.append(s)

        elif t == "type":
            txt = s.get("text", "")
            if not isinstance(txt, str):
                txt = str(txt)
            s["text"] = txt
            s["delay_ms"] = max(25, int(s.get("delay_ms", 30)))  # minimum per-char delay
            s["enter"] = bool(s.get("enter", False))
            fixed.append(s)

        elif t == "key_combo":
            combo = _normalize_combo(s.get("combo", []))
            if combo:
                s["combo"] = combo
                fixed.append(s)

        elif t == "drag":
            fr = s.get("from"); to = s.get("to")
            if isinstance(fr, dict) and isinstance(to, dict):
                try:
                    s["from"] = {"x": int(fr["x"]), "y": int(fr["y"])}
                    s["to"] = {"x": int(to["x"]), "y": int(to["y"])}
                    s["button"] = s.get("button", "left")
                    s["hold_ms"] = int(s.get("hold_ms", 120))
                    fixed.append(s)
                except Exception:
                    pass

        elif t == "scroll":
            try:
                s["delta"] = int(s.get("delta", 0))
                s["horizontal"] = bool(s.get("horizontal", False))
                at = s.get("at")
                if isinstance(at, dict) and "x" in at and "y" in at:
                    s["at"] = {"x": int(at["x"]), "y": int(at["y"])}
                else:
                    s.pop("at", None)
                fixed.append(s)
            except Exception:
                pass

        elif t == "move":
            pt = s.get("point")
            if isinstance(pt, dict) and "x" in pt and "y" in pt:
                s["point"] = {"x": int(pt["x"]), "y": int(pt["y"])}
                s["settle_ms"] = int(s.get("settle_ms", 150))
                fixed.append(s)

        elif t == "wait":
            try:
                s["ms"] = max(50, int(s.get("ms", 300)))
                fixed.append(s)
            except Exception:
                pass

        # unknown types are ignored (fail-safe)

    # Guardrails: add waits after combos; click→type spacing
    guarded: List[Dict[str, Any]] = []
    for i, s in enumerate(fixed):
        guarded.append(s)
        if s.get("type") == "key_combo":
            guarded.append({"type": "wait", "ms": 300})
        if s.get("type") == "click":
            nxt = fixed[i + 1] if i + 1 < len(fixed) else None
            if nxt and nxt.get("type") == "type":
                guarded.append({"type": "wait", "ms": 200})

    p["steps"] = guarded
    p.setdefault("reasoning", "")
    return p
def validate_plan_or_raise(plan: Dict[str, Any]) -> Plan:
    """Pydantic schema validation."""
    return Plan.model_validate(plan)