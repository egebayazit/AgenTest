# filter_utils.py
# v1 filtering for SUT state (no ODS)
# Goals:
# - shrink element list for LLM
# - keep actionable/visible items
# - dedup similar nodes
# - (optional) bias ordering with a simple hint

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import math

# --- basic helpers ---

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
    if t in ACTIONABLE_TYPES:
        return True
    # allow unlabeled Edit fields (often actionable)
    if t == "Edit":
        return True
    # fallback: named item with reasonable rect
    return bool(name)

def _score_element(e: Dict[str, Any], hint: Optional[str]) -> float:
    # simple scoring: enabled + active window + name length + hint keyword hits
    score = 0.0
    if e.get("enabled"): score += 2.0
    if e.get("windowActive"): score += 2.0
    name = (e.get("name") or "").lower()
    path = " > ".join((e.get("path") or []))[:200].lower()

    score += min(len(name)/20.0, 2.0)  # cap name contribution
    if hint:
        h = hint.lower()
        # naive token match
        hits = sum(1 for tok in h.split() if tok and (tok in name or tok in path))
        score += min(hits * 0.8, 3.0)

    # slightly prefer center-ish items (less tiny/edge noise)
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
                # prefer enabled/active with longer name
                cand = max([e, k], key=lambda x: (x.get("enabled"), x.get("windowActive"), len((x.get("name") or ""))))
                # replace if better
                if cand is e:
                    kept.remove(k)
                    kept.append(e)
                is_dup = True
                break
        if not is_dup:
            kept.append(e)
    return kept

# --- public API ---

def filter_pipeline_v1(state: Dict[str, Any], hint: Optional[str] = None, top_n: int = 200) -> Dict[str, Any]:
    """Apply v1 filters to SUT state and return a trimmed version."""
    screen = state.get("screen") or {}
    elems = state.get("elements") or []
    if not isinstance(elems, list):
        elems = []

    # 1) drop invalid rects and tiny items
    pool = [e for e in elems if _valid_rect(e)]

    # 2) actionable-only
    pool = [e for e in pool if _actionable(e)]

    # 3) prefer active window
    active = [e for e in pool if e.get("windowActive")]
    pool = active if active else pool

    # 4) dedup by (name + IoU)
    pool = _dedup(pool, iou_thr=0.85)

    # 5) enabled first
    enabled = [e for e in pool if e.get("enabled")]
    disabled = [e for e in pool if not e.get("enabled")]
    pool = enabled + disabled

    # 6) score & sort
    scored = [(e, _score_element(e, hint)) for e in pool]
    scored.sort(key=lambda t: t[1], reverse=True)
    pool = [e for e, _ in scored][:top_n]

    # 7) return with meta
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

def guard_action_plan_v1(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Light guardrails for /action:
    - limit number of steps
    - ensure wait bounds
    - prevent insane scroll/drag deltas
    - TODO: deep schema validation if needed
    """
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise ValueError("plan.steps must be an array")

    MAX_STEPS = 50
    if len(steps) > MAX_STEPS:
        raise ValueError(f"too many steps ({len(steps)} > {MAX_STEPS})")

    safe = []
    for s in steps:
        t = s.get("type")
        if t == "wait":
            ms = max(50, min(int(s.get("ms", 300)), 5000))
            s["ms"] = ms
        elif t == "scroll":
            delta = int(s.get("delta", 0))
            s["delta"] = max(-1200, min(delta, 1200))
        elif t == "drag":
            # (basic) allow as-is; could clamp coordinates here if needed
            pass
        # keep others
        safe.append(s)

    plan["steps"] = safe
    return plan