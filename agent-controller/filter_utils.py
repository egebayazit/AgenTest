# filter_utils.py
# v1 Controller utilities (no ODS):
# - State filtering for LLM
# - Plan sanitize + validate + guardrails before forwarding to SUT

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Literal
import math
import time
from pydantic import BaseModel, Field, ValidationError

# =========================
# State filtering (v1)
# =========================

ACTIONABLE_TYPES = {
    "Button", "MenuItem", "Hyperlink", "Edit", "ComboBox", "ListItem",
    "TabItem", "RadioButton", "CheckBox", "TreeItem"
}

def _area(rect: Dict[str, int]) -> int:
    return max(0, int(rect.get("w", 0))) * max(0, int(rect.get("h", 0)))

def _iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    ax1, ay1 = a.get("x", 0), a.get("y", 0)
    ax2, ay2 = ax1 + a.get("w", 0), ay1 + a.get("h", 0)
    bx1, by1 = b.get("x", 0), b.get("y", 0)
    bx2, by2 = bx1 + b.get("w", 0), by1 + b.get("h", 0)
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = _area(a) + _area(b) - inter
    return (inter / union) if union > 0 else 0.0

def _valid_rect(e: Dict[str, Any]) -> bool:
    r = e.get("rect") or {}
    return all(k in r for k in ("x", "y", "w", "h")) and _area(r) >= 16*16

def _actionable(e: Dict[str, Any]) -> bool:
    t = (e.get("controlType") or "").strip()
    name = (e.get("name") or "").strip()
    if t in ACTIONABLE_TYPES: return True
    if t == "Edit": return True
    return bool(name)

def _score_element(e: Dict[str, Any], hint: Optional[str]) -> float:
    score = 0.0
    if e.get("enabled"): score += 2.0
    if e.get("windowActive"): score += 2.0
    name = (e.get("name") or "").lower()
    path = " > ".join((e.get("path") or []))[:200].lower()
    score += min(len(name)/20.0, 2.0)
    if hint:
        h = hint.lower()
        hits = sum(1 for tok in h.split() if tok and (tok in name or tok in path))
        score += min(hits * 0.8, 3.0)
    r = e.get("rect") or {}
    area = _area(r)
    score += min(math.log1p(area) / 6.0, 2.0)
    return score

def _dedup(elements: List[Dict[str, Any]], iou_thr: float = 0.85) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for e in elements:
        r = e.get("rect") or {}
        is_dup = False
        for k in kept:
            if _iou(r, k.get("rect") or {}) >= iou_thr and (e.get("name") or "").lower() == (k.get("name") or "").lower():
                cand = max([e, k], key=lambda x: (x.get("enabled"), x.get("windowActive"), len((x.get("name") or ""))))
                if cand is e:
                    kept.remove(k); kept.append(e)
                is_dup = True
                break
        if not is_dup:
            kept.append(e)
    return kept

def filter_pipeline_v1(state: Dict[str, Any], hint: Optional[str] = None, top_n: int = 200) -> Dict[str, Any]:
    screen = state.get("screen") or {}
    elems = state.get("elements") or []
    if not isinstance(elems, list):
        elems = []

    pool = [e for e in elems if _valid_rect(e)]
    pool = [e for e in pool if _actionable(e)]
    active = [e for e in pool if e.get("windowActive")]
    pool = active if active else pool
    pool = _dedup(pool, iou_thr=0.85)

    enabled = [e for e in pool if e.get("enabled")]
    disabled = [e for e in pool if not e.get("enabled")]
    pool = enabled + disabled

    scored = [(e, _score_element(e, hint)) for e in pool]
    scored.sort(key=lambda t: t[1], reverse=True)
    pool = [e for e, _ in scored][:top_n]

    out = {
        "screen": {k: screen.get(k) for k in ("w", "h", "dpiX", "dpiY") if k in screen},
        "elements": pool,
        "meta": {
            "filtered": True,
            "total_in": len(elems),
            "total_out": len(pool),
            "filters": ["validRect>=16x16", "actionable", "activeWindowPref", "dedup(iou>=0.85)", "enabledFirst", "scoreSort", f"top{top_n}"],
            "hint_used": bool(hint),
        },
    }
    return out

# =========================
# Plan sanitize + validate + guard
# =========================

# ---- Allowed keys for key_combo ----
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

# ---- Schema models (Pydantic) ----
class Rect(BaseModel):
    x: int; y: int; w: int; h: int

class ClickTarget(BaseModel):
    rect: Rect

class Click(BaseModel):
    type: Literal["click"]
    button: Literal["left", "right", "middle"] = "left"
    click_count: int = 1
    modifiers: List[str] = []
    target: ClickTarget

class TypeAct(BaseModel):
    type: Literal["type"]
    text: str
    delay_ms: int = 30
    enter: bool = False

class KeyCombo(BaseModel):
    type: Literal["key_combo"]
    combo: List[str]

class Drag(BaseModel):
    type: Literal["drag"]
    from_: Dict[str, int] = Field(..., alias="from")
    to: Dict[str, int]
    button: Literal["left", "right", "middle"] = "left"
    hold_ms: int = 120

class Scroll(BaseModel):
    type: Literal["scroll"]
    delta: int
    horizontal: bool = False
    at: Optional[Dict[str, int]] = None

class Move(BaseModel):
    type: Literal["move"]
    point: Dict[str, int]
    settle_ms: int = 150

class Wait(BaseModel):
    type: Literal["wait"]
    ms: int

Action = Click | TypeAct | KeyCombo | Drag | Scroll | Move | Wait

class Plan(BaseModel):
    action_id: str
    coords_space: Literal["physical"]
    steps: List[Action]
    reasoning: str

def _normalize_combo(keys: Any) -> List[str]:
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        return []
    norm: List[str] = []
    for k in keys:
        if not isinstance(k, str): continue
        kk = _SYNONYMS.get(k.lower().strip(), k.lower().strip())
        if kk in _ALLOWED_KEYS:
            norm.append(kk)
    seen = set(); out: List[str] = []
    for k in norm:
        if k not in seen:
            out.append(k); seen.add(k)
    return out

def sanitize_plan_dict(plan: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(plan)
    p.setdefault("action_id", f"step_{int(time.time())}")
    p.setdefault("coords_space", "physical")

    steps = p.get("steps")
    if not isinstance(steps, list):
        steps = []
    fixed: List[Dict[str, Any]] = []

    for s in steps:
        if not isinstance(s, dict): continue
        t = s.get("type")
        if t == "click":
            tgt = s.get("target", {})
            rect = tgt.get("rect") if isinstance(tgt, dict) else None
            if not isinstance(rect, dict): continue
            try:
                rect = {"x": int(rect["x"]),"y": int(rect["y"]),
                        "w": max(1, int(rect["w"])),"h": max(1, int(rect["h"]))}
            except Exception:
                continue
            s["button"] = s.get("button", "left")
            s["click_count"] = int(s.get("click_count", 1))
            s["modifiers"] = [m.lower() for m in s.get("modifiers", []) if isinstance(m, str)]
            s["target"] = {"rect": rect}
            fixed.append(s)
        elif t == "type":
            txt = s.get("text", "")
            if not isinstance(txt, str): txt = str(txt)
            s["text"] = txt
            s["delay_ms"] = max(25, int(s.get("delay_ms", 30)))
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
        # unknown types -> drop

    # Guardrails: wait after combo; click→type spacer
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
    return Plan.model_validate(plan)

def guard_action_plan_v1(plan: Dict[str, Any]) -> Dict[str, Any]:
    # light runtime bounds
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("plan.steps must be an array")
    if len(steps) > 50:
        raise ValueError(f"too many steps ({len(steps)} > 50)")

    safe = []
    for s in steps:
        t = s.get("type")
        if t == "wait":
            ms = max(50, min(int(s.get("ms", 300)), 5000))
            s["ms"] = ms
        elif t == "scroll":
            delta = int(s.get("delta", 0))
            s["delta"] = max(-1200, min(delta, 1200))
        safe.append(s)
    plan["steps"] = safe
    return plan

def process_plan_v1(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Controller-side single entrypoint:
    LLM raw plan -> sanitize -> validate -> guard -> safe plan
    """
    sanitized = sanitize_plan_dict(plan)
    validate_plan_or_raise(sanitized)
    return guard_action_plan_v1(sanitized)
