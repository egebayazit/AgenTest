# controller_service.py
from __future__ import annotations

import os, re, io, base64, time, unicodedata, hashlib
from typing import Any, Dict, List, Tuple, Optional
import concurrent.futures as cf
import multiprocessing

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- optional: load .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- OCR deps ---
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# --- YOLO (Ultralytics) ---
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False

# -------------------- Config --------------------
SUT_STATE_URL = os.getenv("SUT_STATE_URL", "http://127.0.0.1:18080/state")
LLM_RUN_URL = os.getenv("LLM_RUN_URL")
SUT_TIMEOUT_SEC = int(os.getenv("SUT_TIMEOUT_SEC", "45"))

DEDUPE_KEY_MODE = os.getenv("DEDUPE_KEY_MODE", "name").lower()

# OCR config
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") not in ("0", "false", "False", "")
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")

OCR_FAST_MODE = os.getenv("OCR_FAST_MODE", "1") in ("1","true","True")
OCR_SYNC_BUDGET = int(os.getenv("OCR_SYNC_BUDGET", "12"))

OCR_MAX_AREA_FRAC = float(os.getenv("OCR_MAX_AREA_FRAC", "0.95"))
OCR_MIN_CHAR = int(os.getenv("OCR_MIN_CHAR", "2"))
OCR_MAX_CHAR = int(os.getenv("OCR_MAX_CHAR", "200"))

OCR_MIN_CONF_EMPTY = int(os.getenv("OCR_MIN_CONF_EMPTY", "22"))
OCR_MIN_CONF_DUP   = int(os.getenv("OCR_MIN_CONF_DUP", "28"))
OCR_MIN_CONF_SHORT = int(os.getenv("OCR_MIN_CONF_SHORT", "40"))
OCR_ACCEPT_LOWCONF_MIN = int(os.getenv("OCR_ACCEPT_LOWCONF_MIN", "12"))

OCR_EARLY_STOP_DELTA = int(os.getenv("OCR_EARLY_STOP_DELTA", "12"))
OCR_EARLY_STOP_ABS   = int(os.getenv("OCR_EARLY_STOP_ABS", "85"))

RECT_PAD_PX = int(os.getenv("RECT_PAD_PX", "3"))

OCR_USE_HARD_FILTERS = os.getenv("OCR_USE_HARD_FILTERS", "0") in ("1","true","True")
THIN_LINE_HARD_PX = int(os.getenv("THIN_LINE_HARD_PX", "4"))
THIN_LINE_SOFT_W  = int(os.getenv("THIN_LINE_SOFT_W", "600"))
THIN_LINE_SOFT_H  = int(os.getenv("THIN_LINE_SOFT_H", "8"))

TRY_INVERT = os.getenv("TRY_INVERT", "0") in ("1","true","True")
FALLBACK_LANG = (os.getenv("FALLBACK_LANG") or "").strip()

OCR_CHAR_WHITELIST = os.getenv(
    "OCR_CHAR_WHITELIST",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ğĞüÜşŞıİöÖçÇ-_.:/@#()[]{}+&%!?,'\" "
)
OCR_STRICT_WHITELIST = os.getenv("OCR_STRICT_WHITELIST", "0") in ("1","true","True")

# Hız ayarları
OCR_COOLDOWN_SEC = int(os.getenv("OCR_COOLDOWN_SEC", "6"))
_PAR = int(os.getenv("OCR_PARALLEL", "0"))
if _PAR <= 0:
    try:
        _PAR = max(2, min(8, multiprocessing.cpu_count()))
    except Exception:
        _PAR = 4

# --- YOLO (Icon) ---
ICON_YOLO_ENABLED = os.getenv("ICON_YOLO_ENABLED", "1") in ("1","true","True")
ICON_YOLO_MODEL_PATH = os.getenv("ICON_YOLO_MODEL_PATH", "").strip()
ICON_YOLO_CONF = float(os.getenv("ICON_YOLO_CONF", "0.35"))
ICON_YOLO_IOU = float(os.getenv("ICON_YOLO_IOU", "0.5"))
ICON_MATCH_MIN_IOU = float(os.getenv("ICON_MATCH_MIN_IOU", "0.30"))
ICON_ONLY_ON_EMPTY = os.getenv("ICON_ONLY_ON_EMPTY", "1") in ("1","true","True")
ICON_DETECT_WHOLE_SCREEN = os.getenv("ICON_DETECT_WHOLE_SCREEN", "1") in ("1","true","True")
ICON_BOX_MIN_W = int(os.getenv("ICON_BOX_MIN_W", "10"))
ICON_BOX_MIN_H = int(os.getenv("ICON_BOX_MIN_H", "10"))
ICON_BOX_MAX_W = int(os.getenv("ICON_BOX_MAX_W", "256"))
ICON_BOX_MAX_H = int(os.getenv("ICON_BOX_MAX_H", "256"))
ICON_CLASS_OVERRIDES = (os.getenv("ICON_CLASS_OVERRIDES") or "").strip()

# İkon sınıf adlarını LLM için normalize et (semantic label)
ICON_NAME_SEMANTICS: Dict[str, str] = {
    "View": "container",
    "ImageView": "icon",
    "Text": "text",
    "Line": "separator",
}

APP_NAME = "AgenTest Controller"
APP_VER = "0.17.0-jvm-normalize+llm-minimal"

STRIP_FIELDS = {"controlType", "enabled", "patterns", "idx"}

app = FastAPI(title=APP_NAME, version=APP_VER)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# caches
app.state.last_raw_state: Optional[Dict[str, Any]] = None
app.state.last_raw_ts: Optional[float] = None
app.state.last_sut_error: Optional[Dict[str, Any]] = None

app.state.ocr_queue: List[Dict[str, Any]] = []
app.state.ocr_queue_max = 500

# OCR caches
_OCR_CACHE: Dict[Tuple, Tuple[str, float]] = {}
_OCR_CACHE_MAX = 800
_PREPROC_CACHE: Dict[Tuple, Image.Image] = {}
_PREPROC_MAX = 400
_COOLDOWN: Dict[Tuple, float] = {}

# YOLO model singletons
_YOLO_MODEL: Optional[YOLO] = None
_YOLO_CLASS_NAMES: Dict[int, str] = {}

# -------------------- http utils --------------------
def _httpx_timeout() -> httpx.Timeout:
    return httpx.Timeout(SUT_TIMEOUT_SEC, read=SUT_TIMEOUT_SEC, connect=5.0)

async def _fetch_raw_state() -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=_httpx_timeout(), trust_env=False) as client:
            r = await client.post(SUT_STATE_URL, json={})
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        app.state.last_sut_error = {
            "type": type(e).__name__,
            "url": SUT_STATE_URL,
            "message": str(e),
        }
        raise HTTPException(status_code=502, detail=f"SUT /state failed: {app.state.last_sut_error}")
    app.state.last_raw_state = data
    app.state.last_raw_ts = time.time()
    app.state.last_sut_error = None
    return data

# -------------------- geometry --------------------
def _rect_area(el: Dict[str, Any]) -> int:
    r = (el.get("rect") or {})
    w = max(0, int(r.get("r", 0)) - int(r.get("l", 0)))
    h = max(0, int(r.get("b", 0)) - int(r.get("t", 0)))
    return w * h

def _screen_size(raw: Dict[str, Any]) -> Tuple[int, int]:
    scr = raw.get("screen") or {}
    return int(scr.get("w", 0)), int(scr.get("h", 0))

def _iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    ax1, ay1, ax2, ay2 = int(a["l"]), int(a["t"]), int(a["r"]), int(a["b"])
    bx1, by1, bx2, by2 = int(b["l"]), int(b["t"]), int(b["r"]), int(b["b"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = max(1, area_a + area_b - inter)
    return inter / union

# -------------------- names --------------------
_ZW_REMOVE = {"Cf"}
_ws_re = re.compile(r"\s+", re.UNICODE)

def _canonical_name(name: str) -> str:
    s = (name or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in _ZW_REMOVE)
    s = _ws_re.sub(" ", s)
    return s.casefold()

def _effective_name(el: Dict[str, Any]) -> str:
    raw = (el.get("name") or "").strip()
    return raw if raw else (el.get("name_ocr") or "").strip()

def _canonical_effective_name(el: Dict[str, Any]) -> str:
    return _canonical_name(_effective_name(el))

def _group_key_for(el: Dict[str, Any]) -> tuple:
    cname = _canonical_effective_name(el)
    if DEDUPE_KEY_MODE == "name_active":
        return (cname, bool(el.get("windowActive")))
    return (cname,)

_CT_PRIORITY = {
    "Text": 1, "Edit": 2, "Button": 3, "Hyperlink": 4, "ListItem": 5, "TabItem": 6,
    "ComboBox": 7, "MenuItem": 8, "Window": 9, "Other": 10
}
def _ct_rank(el: Dict[str, Any]) -> int:
    ct = (el.get("controlType") or "Other")
    return _CT_PRIORITY.get(ct, 10)

# -------------------- OCR queue helpers --------------------
def _enqueue_ocr(items: List[Dict[str, Any]], reason: str, key: tuple) -> None:
    if not items:
        return
    ts = time.time()
    to_add = [{
        "ts": ts,
        "reason": reason,
        "key": str(key),
        "name": it.get("name"),
        "rect": it.get("rect"),
        "center": it.get("center"),
        "path": it.get("path"),
        "windowActive": it.get("windowActive", None),
        "controlType": it.get("controlType", None),
    } for it in items]
    app.state.ocr_queue.extend(to_add)
    overflow = max(0, len(app.state.ocr_queue) - app.state.ocr_queue_max)
    if overflow:
        del app.state.ocr_queue[0:overflow]

# -------------------- screenshot & crops --------------------
def _screenshot_from_raw(raw: Dict[str, Any]) -> Tuple[Optional[Image.Image], Optional[str]]:
    if not isinstance(raw, dict):
        return None, None
    b64 = None
    sc = raw.get("screenshot")
    if isinstance(sc, dict):
        b64 = sc.get("b64") or sc.get("image_b64") or sc.get("image") or sc.get("data")
        if isinstance(b64, str) and len(b64) < 1000:
            b64 = None
    elif isinstance(sc, str) and len(sc) > 1000:
        b64 = sc
    if not b64:
        for k in ("screenshot_base64", "screenshot_png", "screenshot_jpg", "b64", "image_b64", "image"):
            v = raw.get(k)
            if isinstance(v, str) and len(v) > 1000:
                b64 = v
                break
    if not b64:
        return None, None
    try:
        data = base64.b64decode(b64, validate=False)
        md5 = hashlib.md5(data).hexdigest()
        return Image.open(io.BytesIO(data)).convert("RGB"), md5
    except Exception:
        return None, None

def _too_big_rect(rect: Dict[str, Any], scr_w: int, scr_h: int) -> bool:
    if not (scr_w and scr_h): return False
    l, t = max(0, int(rect.get("l", 0))), max(0, int(rect.get("t", 0)))
    r, b = int(rect.get("r", 0)), int(rect.get("b", 0))
    r = max(l + 1, r); b = max(t + 1, b)
    area = (r - l) * (b - t)
    return area > OCR_MAX_AREA_FRAC * scr_w * scr_h

def _too_thin_line(rect: Dict[str, Any], scr_w: int) -> bool:
    h = max(0, int(rect.get("b", 0)) - int(rect.get("t", 0)))
    w = max(0, int(rect.get("r", 0)) - int(rect.get("l", 0)))
    if h <= THIN_LINE_HARD_PX: return True
    if h <= THIN_LINE_SOFT_H and w > max(THIN_LINE_SOFT_W, scr_w // 2): return True
    return False

def _pad_and_crop(img: Image.Image, rect: Dict[str, Any]) -> Image.Image:
    l = max(0, int(rect.get("l", 0)))
    t = max(0, int(rect.get("t", 0)))
    r = max(l + 1, int(rect.get("r", 0)))
    b = max(t + 1, int(rect.get("b", 0)))
    if RECT_PAD_PX > 0:
        l = max(0, l - RECT_PAD_PX); t = max(0, t - RECT_PAD_PX)
        r = min(img.width,  r + RECT_PAD_PX); b = min(img.height, b + RECT_PAD_PX)
    return img.crop((l, t, r, b))

# mojibake fix & post-normalize
_MOJI_FIXES = {
    "Ã§":"ç","Ã‡":"Ç","Ã¶":"ö","Ã–":"Ö","Ã¼":"ü","Ãœ":"Ü",
    "Ä±":"ı","Ä°":"İ","ÅŸ":"ş","Åž":"Ş","ÄŸ":"ğ","Äž":"Ğ",
    "Â©":"©","Â®":"®","Â·":"·","Â«":"«","Â»":"»","Â":"",
}
_lead_junk_re = re.compile(r"^[^\wğĞüÜşŞıİöÖçÇ]+")
_many_seps_re = re.compile(r"(\|\s*){3,}")
_menu_pattern_re = re.compile(r"([A-Z][a-z]+){4,}")

def _post_normalize(s: str) -> str:
    if not s: return s
    out = s
    for k,v in _MOJI_FIXES.items():
        if k in out: out = out.replace(k,v)
    out = _lead_junk_re.sub("", out).strip()
    out = re.sub(r"\s+", " ", out)
    out = _many_seps_re.sub("|", out)
    if _menu_pattern_re.search(out) and len(out) > 30:
        words = out.split()
        if words and len(words[0]) < 20:
            out = words[0]
    return out

# -------------------- OCR core --------------------
def _ocr_try(crop: Image.Image, shash: str, lang: str, psm: int, cfg_extra: str = "") -> Tuple[str, float, bool]:
    cfg = f"--psm {psm}" + (f" {cfg_extra}" if cfg_extra else "")
    if OCR_CHAR_WHITELIST:
        cfg += f' -c tessedit_char_whitelist="{OCR_CHAR_WHITELIST}"'
        if OCR_STRICT_WHITELIST:
            cfg += " -c classify_bln_numeric_mode=1"
    try:
        data = pytesseract.image_to_data(crop, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
        words = data.get("text", []) or []
        confs = data.get("conf", []) or []
        pairs = [(w.strip(), float(c)) for w, c in zip(words, confs) if w and w.strip() and float(c) >= 0]
        if not pairs:
            text, avg_conf = "", 0.0
        else:
            text = " ".join(w for w, _ in pairs)
            avg_conf = sum(c for _, c in pairs) / max(1, len(pairs))
        text = _post_normalize(text)
        return text, avg_conf, False
    except Exception:
        return "", 0.0, False

def _best_psm_candidates(rect: Dict[str, Any]) -> List[int]:
    h = max(1, int(rect.get("b", 0)) - int(rect.get("t", 0)))
    w = max(1, int(rect.get("r", 0)) - int(rect.get("l", 0)))
    single_line = (h <= 40) or (w >= 4*h)
    if OCR_FAST_MODE:
        return [7 if single_line else 6, 11]
    primary = 7 if single_line else 6
    order = [primary, (6 if primary == 7 else 7), 11, 13]
    seen, uniq = set(), []
    for x in order:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def _ocr_with_quality(img: Image.Image, shash: str, rect: Dict[str, Any],
                      lang_main: str, lang_fallback: str, psm_list: List[int],
                      min_conf_target: float) -> Tuple[str, float, int, str, str, bool, bool]:
    base_key = (shash, rect.get("l"), rect.get("t"), rect.get("r"), rect.get("b"), RECT_PAD_PX)

    # cooldown
    now = time.time()
    if (base_key in _COOLDOWN) and (now - _COOLDOWN[base_key] < OCR_COOLDOWN_SEC):
        return "", 0.0, psm_list[0], lang_main, "cooldown", True, True
    _COOLDOWN[base_key] = now

    def _get_pp(invert: bool) -> Image.Image:
        k = base_key + (invert,)
        if k in _PREPROC_CACHE:
            return _PREPROC_CACHE[k]
        crop = _pad_and_crop(img, rect)
        g = ImageOps.grayscale(crop)
        if invert: g = ImageOps.invert(g)
        g = g.resize((max(1, int(g.width * 2.0)), max(1, int(g.height * 2.0))))
        g = ImageOps.autocontrast(g)
        g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))
        try:
            hist = g.histogram()
            mean = max(1, sum(i * c for i, c in enumerate(hist)) // max(1, sum(hist)))
        except Exception:
            mean = 128
        g = g.point(lambda p: 255 if p > mean else 0)
        if len(_PREPROC_CACHE) > _PREPROC_MAX:
            for _ in range(32):
                try:
                    _PREPROC_CACHE.pop(next(iter(_PREPROC_CACHE)))
                except Exception:
                    break
        _PREPROC_CACHE[k] = g
        return g

    best = ("", 0.0, (psm_list[0] if psm_list else 6), lang_main, "normal", False, False)

    def should_stop(conf: float) -> bool:
        return (conf >= OCR_EARLY_STOP_ABS) or (conf >= (min_conf_target + OCR_EARLY_STOP_DELTA))

    def try_langs(pp: Image.Image, source: str) -> bool:
        nonlocal best
        langs = [lang_main] if OCR_FAST_MODE or not lang_fallback else [lang_main, lang_fallback]
        for psm in psm_list:
            for lang in langs:
                ck = base_key + (lang, psm, source)
                if ck in _OCR_CACHE:
                    t, c = _OCR_CACHE[ck]
                    cache_hit = True
                else:
                    t, c, cache_hit = _ocr_try(pp, shash, lang, psm, cfg_extra=("--oem 1" if lang != lang_main else ""))
                    if len(_OCR_CACHE) > _OCR_CACHE_MAX:
                        for _ in range(64):
                            try:
                                _OCR_CACHE.pop(next(iter(_OCR_CACHE)))
                            except Exception:
                                break
                    _OCR_CACHE[ck] = (t, c)
                if c > best[1]:
                    best = (t, c, psm, lang, source, best[5], cache_hit)
                if should_stop(c):
                    best = (best[0], best[1], psm, lang, source, True, best[6])
                    return True
        return False

    if try_langs(_get_pp(False), "normal"):
        return best
    if TRY_INVERT:
        try_langs(_get_pp(True), "invert")
    return best

def _enrich_with_ocr_if_possible(raw: Dict[str, Any],
                                 elements: List[Dict[str, Any]],
                                 dbg: Dict[str, Any] | None = None) -> None:
    if not OCR_ENABLED:
        if dbg is not None:
            dbg["ocr_sync_done"] = 0
            dbg["ocr_sync_skipped"] = "disabled"
        return

    img, shash = _screenshot_from_raw(raw)
    if img is None or not shash:
        if dbg is not None:
            dbg["ocr_sync_done"] = 0
            dbg["ocr_sync_skipped"] = "no_screenshot_in_state"
        return

    scr_w, scr_h = _screen_size(raw)

    empty_effective = [el for el in elements if not _effective_name(el)]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for el in elements:
        eff = _canonical_effective_name(el)
        if eff:
            groups.setdefault(eff, []).append(el)
    dup_elems: List[Dict[str, Any]] = []
    for _, grp in groups.items():
        if len(grp) > 1:
            dup_elems.extend(grp)

    candidates = empty_effective + dup_elems

    def _prio_tuple(el: Dict[str, Any]):
        return (_ct_rank(el), _rect_area(el))
    candidates.sort(key=_prio_tuple)

    workset = []
    for el in candidates:
        if len(workset) >= OCR_SYNC_BUDGET: break
        r = el.get("rect")
        if not r: continue
        if OCR_USE_HARD_FILTERS:
            if _too_big_rect(r, scr_w, scr_h): continue
            if _too_thin_line(r, scr_w): continue
        workset.append(el)

    sync_done = 0
    attempted = 0
    rejected_empty = 0
    accepted_high = 0
    accepted_low = 0
    tried_psm = 0
    early_stops = 0
    cache_hits = 0
    cooldown_skips = 0

    def _process_el(el: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r = el.get("rect") or {}
        psm_list = _best_psm_candidates(r)
        is_empty = not (el.get("name") or "").strip()
        min_conf = OCR_MIN_CONF_EMPTY if is_empty else OCR_MIN_CONF_DUP

        text, avg_conf, used_psm, used_lang, source, early, cache_hit = _ocr_with_quality(
            img, shash, r, TESSERACT_LANG,
            ("" if OCR_FAST_MODE else FALLBACK_LANG),
            psm_list, min_conf
        )

        stats = {
            "attempted": 1,
            "tried_psm": len(psm_list),
            "early": 1 if early else 0,
            "cache_hit": 1 if cache_hit else 0,
            "cooldown": 1 if source == "cooldown" else 0,
            "rejected_empty": 0,
            "accepted_high": 0,
            "accepted_low": 0,
            "sync_done": 0,
        }

        if not text:
            stats["rejected_empty"] = 1
            return {}, stats

        updates: Dict[str, Any] = {}
        if (len(text) >= OCR_MIN_CHAR) and (len(text) <= OCR_MAX_CHAR) and (avg_conf >= min_conf):
            if is_empty:
                updates["name_ocr"] = text
            else:
                arr = el.get("alt_names_ocr")
                if not isinstance(arr, list): arr = []
                if text not in arr: arr.append(text)
                if len(arr) > 3: arr = arr[-3:]
                updates["alt_names_ocr"] = arr
            updates.update({
                "name_ocr_conf": avg_conf,
                "name_ocr_psm": used_psm,
                "name_ocr_lang": used_lang,
                "name_ocr_src": source,
                "name_ocr_quality": "high",
            })
            stats["accepted_high"] = 1
            stats["sync_done"] = 1
            return updates, stats

        if avg_conf >= OCR_ACCEPT_LOWCONF_MIN:
            updates.update({
                "name_ocr_tentative": text,
                "name_ocr_conf": avg_conf,
                "name_ocr_psm": used_psm,
                "name_ocr_lang": used_lang,
                "name_ocr_src": source,
                "name_ocr_quality": "low",
            })
            stats["accepted_low"] = 1
            stats["sync_done"] = 1
            return updates, stats

        stats["rejected_empty"] = 1
        return {}, stats

    with cf.ThreadPoolExecutor(max_workers=_PAR) as ex:
        results = list(ex.map(_process_el, workset))

    for el, (updates, st) in zip(workset, results):
        attempted += st["attempted"]
        tried_psm += st["tried_psm"]
        early_stops += st["early"]
        cache_hits += st["cache_hit"]
        cooldown_skips += st["cooldown"]
        rejected_empty += st["rejected_empty"]
        accepted_high += st["accepted_high"]
        accepted_low += st["accepted_low"]
        sync_done += st["sync_done"]
        if updates:
            el.update(updates)

    if dbg is not None:
        dbg.update({
            "ocr_sync_done": sync_done,
            "ocr_attempted": attempted,
            "ocr_high": accepted_high,
            "ocr_low": accepted_low,
            "ocr_early_stops": early_stops,
            "ocr_cache_hits": cache_hits,
            "ocr_cooldown_skipped": cooldown_skips,
            "ocr_sync_pending": max(0, len(candidates) - len(workset)),
            "ocr_rejected_empty": rejected_empty,
            "ocr_tried_psm": tried_psm,
            "ocr_parallel": _PAR,
        })

# -------------------- YOLO Icon Detection --------------------
def _yolo_is_ready() -> bool:
    return _YOLO_AVAILABLE and bool(ICON_YOLO_MODEL_PATH)

def _load_yolo_once() -> None:
    global _YOLO_MODEL, _YOLO_CLASS_NAMES
    if _YOLO_MODEL is not None:
        return
    if not _yolo_is_ready():
        return
    try:
        _YOLO_MODEL = YOLO(ICON_YOLO_MODEL_PATH)
        class_names = {}
        try:
            names = getattr(_YOLO_MODEL.model, "names", None) or getattr(_YOLO_MODEL, "names", None)
            if isinstance(names, dict):
                class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, list):
                class_names = {i: str(n) for i, n in enumerate(names)}
        except Exception:
            class_names = {}
        if ICON_CLASS_OVERRIDES:
            for token in ICON_CLASS_OVERRIDES.split(","):
                token = token.strip()
                if not token or ":" not in token: continue
                cid_s, cname = token.split(":", 1)
                try:
                    cid = int(cid_s)
                    class_names[cid] = cname.strip()
                except Exception:
                    pass
        _YOLO_CLASS_NAMES = class_names
    except Exception:
        _YOLO_MODEL = None
        _YOLO_CLASS_NAMES = {}

def _cls_name(cid: int) -> str:
    if cid in _YOLO_CLASS_NAMES:
        return _YOLO_CLASS_NAMES[cid]
    return f"class_{cid}"

def _bbox_within_limits(xyxy: Tuple[float,float,float,float]) -> bool:
    x1,y1,x2,y2 = xyxy
    w = max(0, int(x2 - x1))
    h = max(0, int(y2 - y1))
    if w < ICON_BOX_MIN_W or h < ICON_BOX_MIN_H: return False
    if w > ICON_BOX_MAX_W or h > ICON_BOX_MAX_H: return False
    return True

def _yolo_on_image(img: Image.Image):
    if _YOLO_MODEL is None:
        return []
    try:
        res = _YOLO_MODEL.predict(img, conf=ICON_YOLO_CONF, iou=ICON_YOLO_IOU, verbose=False)
        out = []
        if not res:
            return out
        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return out
        xyxy_list = boxes.xyxy.cpu().numpy().tolist()
        cls_list = boxes.cls.cpu().numpy().tolist()
        conf_list = boxes.conf.cpu().numpy().tolist()
        for xyxy, cid, conf in zip(xyxy_list, cls_list, conf_list):
            if not _bbox_within_limits(tuple(xyxy)):
                continue
            out.append({
                "xyxy": xyxy,
                "cls_id": int(cid),
                "cls_name": _cls_name(int(cid)),
                "conf": float(conf),
            })
        return out
    except Exception:
        return []

def _assign_icons_by_overlap(img: Image.Image,
                             elements: List[Dict[str, Any]],
                             dbg: Dict[str, Any] | None = None) -> None:
    dets = _yolo_on_image(img)
    if dbg is not None:
        dbg["icon_yolo"] = {
            "enabled": ICON_YOLO_ENABLED and _yolo_is_ready(),
            "model_loaded": _YOLO_MODEL is not None,
            "detections_total": len(dets),
            "conf": ICON_YOLO_CONF,
            "iou": ICON_YOLO_IOU,
            "match_min_iou": ICON_MATCH_MIN_IOU,
        }
    if not dets:
        return

    targets = []
    for el in elements or []:
        if ICON_ONLY_ON_EMPTY and _effective_name(el):
            continue
        targets.append(el)

    matched = 0
    for el in targets:
        r = el.get("rect") or {}
        if not r: continue
        el_box = {"l": int(r.get("l", 0)), "t": int(r.get("t", 0)), "r": int(r.get("r", 0)), "b": int(r.get("b", 0))}
        best = None
        best_iou = 0.0
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            yb = {"l": int(x1), "t": int(y1), "r": int(x2), "b": int(y2)}
            iou = _iou(el_box, yb)
            if iou > best_iou:
                best_iou = iou
                best = d
        if best and best_iou >= ICON_MATCH_MIN_IOU:
            label = best["cls_name"]
            conf = best["conf"]
            el["name_icon"] = label
            el["name_icon_conf"] = round(conf, 3)
            el["name_icon_box"] = [int(v) for v in best["xyxy"]]
            el["name_icon_iou"] = round(best_iou, 3)

            if not el.get("name") and not el.get("name_ocr"):
                el["name_ocr"] = label
                el["name_ocr_quality"] = "icon"
                el["name_ocr_conf"] = max(40.0, 100.0 * conf)
            else:
                arr = el.get("alt_names_ocr")
                if not isinstance(arr, list): arr = []
                if label not in arr: arr.append(label)
                el["alt_names_ocr"] = arr[-6:]
            matched += 1

    if dbg is not None and "icon_yolo" in dbg:
        dbg["icon_yolo"]["matched"] = matched

def _enrich_with_icons_yolo(raw: Dict[str, Any],
                            elements: List[Dict[str, Any]],
                            dbg: Dict[str, Any] | None = None) -> None:
    if not ICON_YOLO_ENABLED or not _yolo_is_ready():
        if dbg is not None:
            dbg["icon_yolo"] = {"enabled": False, "model_loaded": False}
        return
    _load_yolo_once()
    if _YOLO_MODEL is None:
        if dbg is not None:
            dbg["icon_yolo"] = {"enabled": True, "model_loaded": False}
        return
    img, shash = _screenshot_from_raw(raw)
    if img is None:
        if dbg is not None:
            dbg["icon_yolo"] = {"enabled": True, "model_loaded": True, "skipped": "no_screenshot"}
        return

    if ICON_DETECT_WHOLE_SCREEN:
        _assign_icons_by_overlap(img, elements, dbg)
        return

    targets = []
    for el in elements or []:
        if ICON_ONLY_ON_EMPTY and _effective_name(el):
            continue
        r = el.get("rect") or {}
        if not r: continue
        crop = _pad_and_crop(img, r)
        targets.append((el, crop))

    tried = 0
    matched = 0
    for el, crop in targets:
        tried += 1
        try:
            res = _YOLO_MODEL.predict(crop, conf=ICON_YOLO_CONF, iou=ICON_YOLO_IOU, verbose=False)
            if not res: continue
            r0 = res[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is None or boxes.xyxy.shape[0] == 0:
                continue
            confs = boxes.conf.cpu().numpy().tolist()
            idx = max(range(len(confs)), key=lambda i: confs[i])
            cid = int(boxes.cls.cpu().numpy().tolist()[idx])
            label = _cls_name(cid)
            conf = float(confs[idx])
            el["name_icon"] = label
            el["name_icon_conf"] = round(conf, 3)
            if not el.get("name") and not el.get("name_ocr"):
                el["name_ocr"] = label
                el["name_ocr_quality"] = "icon"
                el["name_ocr_conf"] = max(40.0, 100.0 * conf)
            else:
                arr = el.get("alt_names_ocr")
                if not isinstance(arr, list): arr = []
                if label not in arr: arr.append(label)
                el["alt_names_ocr"] = arr[-6:]
            matched += 1
        except Exception:
            continue

    if dbg is not None:
        dbg["icon_yolo"] = {
            "enabled": True,
            "model_loaded": True,
            "per_crop_mode": True,
            "attempted": tried,
            "matched": matched,
            "conf": ICON_YOLO_CONF,
            "iou": ICON_YOLO_IOU,
        }

# -------------------- JVM → Windows-like normalize --------------------
_JVM_CLASS_CT_MAP = {
    # Rough mapping to keep dedupe/priority happy
    "JButton": "Button",
    "SquareStripeButton": "Button",
    "CloseButton": "Button",
    "JLabel": "Text",
    "TextPanel": "Text",
    "WithIconAndArrows": "Text",
    "JBTextField": "Edit",
    "JTextField": "Edit",
    "JCheckBox": "Button",
    "JRadioButton": "Button",
    "TabItem": "TabItem",
    # Default: Other
}

def _is_jvm_state(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    st = (data.get("state_type") or "").strip().lower()
    if st == "jvm": return True
    if st == "windows": return False
    # Heuristic: JVM dumps often have 'componentCount' and nested 'elements' trees with 'class'/'children'
    if "componentCount" in data and isinstance(data.get("elements"), list):
        return True
    # Windows states typically have 'screen' with w/h and element rects
    if isinstance(data.get("screen"), dict):
        return False
    # If top has 'b64' and deeply nested 'children', lean JVM
    if "b64" in data:
        return True
    return False

def _jvm_flatten(nodes: List[Dict[str, Any]],
                 parent_path: List[str],
                 out: List[Dict[str, Any]],
                 mins: Dict[str, int],
                 maxs: Dict[str, int]) -> None:
    if not nodes: return
    for n in nodes:
        cls = str(n.get("class") or "Unknown")
        text = (n.get("text") or "").strip()
        x = n.get("x"); y = n.get("y"); w = n.get("width"); h = n.get("height")
        ct = _JVM_CLASS_CT_MAP.get(cls, "Other")

        # Build path token (keep short, no huge strings)
        token_label = text if text and len(text) <= 64 else cls
        token = f"{ct}({token_label})" if token_label else f"Other({cls})"
        path_now = (parent_path + [token])[-3:]  # keep last 3 like Windows feel

        if isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            l, t, r, b = x, y, x + w, y + h
            mins["l"] = min(mins["l"], l)
            mins["t"] = min(mins["t"], t)
            maxs["r"] = max(maxs["r"], r)
            maxs["b"] = max(maxs["b"], b)

            out.append({
                "name": text,
                "rect": {"l": l, "t": t, "r": r, "b": b},
                "center": {"x": l + w // 2, "y": t + h // 2},
                "path": path_now,
                "controlType": ct,
                "windowActive": True,  # best-effort; JVM dump genelde tek aktif pencere
            })

        # Recurse
        ch = n.get("children")
        if isinstance(ch, list) and ch:
            _jvm_flatten(ch, path_now, out, mins, maxs)

def _normalize_jvm_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Top-level screenshot
    b64 = raw.get("b64")
    # Entry nodes (the sample uses 'elements' with a tree of 'children')
    roots = raw.get("elements") or []
    flat: List[Dict[str, Any]] = []
    mins = {"l": 1<<30, "t": 1<<30}
    maxs = {"r": 0, "b": 0}
    _jvm_flatten(roots, [], flat, mins, maxs)

    # If coordinates start negative, offset to 0,0
    off_l = 0 if mins["l"] >= 0 else -mins["l"]
    off_t = 0 if mins["t"] >= 0 else -mins["t"]

    for el in flat:
        r = el["rect"]
        r["l"] += off_l; r["t"] += off_t; r["r"] += off_l; r["b"] += off_t
        c = el["center"]
        c["x"] += off_l; c["y"] += off_t

    w = max(0, maxs["r"] + off_l)
    h = max(0, maxs["b"] + off_t)

    norm = {
        "screen": {"w": w, "h": h, "dpiX": 96, "dpiY": 96},
        "elements": flat,
        "timestamp": int(time.time() * 1000),
        "state_type": "JVM",
    }
    if b64 and isinstance(b64, str) and len(b64) > 1000:
        norm["screenshot"] = {"b64": b64}
    return norm

def _maybe_normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if _is_jvm_state(raw):
            return _normalize_jvm_state(raw)
        # If Windows-like but missing timestamp, add a fresh one (non-breaking)
        if "timestamp" not in raw:
            raw = dict(raw)
            raw["timestamp"] = int(time.time() * 1000)
        return raw
    except Exception:
        # Fail-safe: just pass original
        return raw

# -------------------- dedupe --------------------
def _dedupe_by_name_smart(elements: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    dropped_empty = 0
    for el in (elements or []):
        eff = _effective_name(el)
        if not eff:
            dropped_empty += 1
            continue
        groups.setdefault(_group_key_for(el), []).append(el)

    kept: List[Dict[str, Any]] = []
    duplicate_groups = 0
    kept_names_preview: List[str] = []
    ocr_enqueued = 0

    for key, group in groups.items():
        group_sorted = sorted(group, key=_ct_rank)
        best_ct = _ct_rank(group_sorted[0])
        group_ct = [g for g in group_sorted if _ct_rank(g) == best_ct]

        active = [g for g in group_ct if bool(g.get("windowActive"))]
        candidates = active if active else group_ct

        chosen = min(candidates, key=_rect_area)
        kept.append(chosen)
        kept_names_preview.append(_effective_name(chosen) or chosen.get("name", ""))

        leftovers = [g for g in group if g is not chosen]
        if leftovers:
            duplicate_groups += 1
            alt = chosen.get("alt_names_ocr")
            if not isinstance(alt, list):
                alt = []
            for lo in leftovers:
                if lo.get("name_ocr"):
                    if lo["name_ocr"] and lo["name_ocr"] not in alt:
                        alt.append(lo["name_ocr"])
                if lo.get("name_ocr_tentative"):
                    if lo["name_ocr_tentative"] and lo["name_ocr_tentative"] not in alt:
                        alt.append(lo["name_ocr_tentative"])
                lo_alts = lo.get("alt_names_ocr")
                if isinstance(lo_alts, list):
                    for s in lo_alts:
                        if s and s not in alt:
                            alt.append(s)
            if len(alt) > 6:
                alt = alt[-6:]
            if alt:
                chosen["alt_names_ocr"] = alt

            _enqueue_ocr(leftovers, reason="duplicate_dedupe", key=key)
            ocr_enqueued += len(leftovers)

    debug = {
        "dedupe_mode": f"{DEDUPE_KEY_MODE}+ct_rank+canon",
        "dropped_empty_names": dropped_empty,
        "duplicate_groups": duplicate_groups,
        "kept_count": len(kept),
        "kept_names_preview": kept_names_preview[:20],
        "ocr_enqueued": ocr_enqueued,
        "ocr_queue_size": len(app.state.ocr_queue),
    }
    return kept, debug

# -------------------- finalize for LLM --------------------
def _normalized_name_from_icon(el: Dict[str, Any]) -> Optional[str]:
    label = (el.get("name_icon") or "").strip()
    if not label:
        return None
    return ICON_NAME_SEMANTICS.get(label, label)

def _build_llm_element(el: Dict[str, Any], final_name: str) -> Dict[str, Any]:
    # LLM'e sadece name, center, path
    return {
        "name": final_name,
        "center": el.get("center"),
        "path": el.get("path"),
    }

def _finalize_elements_for_llm(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for el in elements or []:
        name = (el.get("name") or "").strip()
        if not name:
            name = (el.get("name_ocr") or "").strip()
        if not name:
            icon_name = _normalized_name_from_icon(el)
            if icon_name:
                name = icon_name
        if not name:
            continue
        out.append(_build_llm_element(el, name))
    return out

# -------------------- strip (for /state/filtered inspection) --------------------
def _strip_fields_for_llm(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for el in elements or []:
        cleaned.append({k: v for k, v in el.items() if k not in STRIP_FIELDS})
    return cleaned

# -------------------- filter pipelines --------------------
def _filter_state_for_llm(raw: Dict[str, Any]) -> Dict[str, Any]:
    proc = _maybe_normalize(raw)
    elems = proc.get("elements") or []
    dbg: Dict[str, Any] = {}
    _enrich_with_ocr_if_possible(proc, elems, dbg)
    _enrich_with_icons_yolo(proc, elems, dbg)
    unique, _debug = _dedupe_by_name_smart(elems)
    final_elems = _finalize_elements_for_llm(unique)
    return {"screen": proc.get("screen"), "timestamp": int(time.time()*1000), "elements": final_elems}

def _filter_state_for_inspect(raw: Dict[str, Any]) -> Dict[str, Any]:
    proc = _maybe_normalize(raw)
    elems = proc.get("elements") or []
    dbg: Dict[str, Any] = {}
    _enrich_with_ocr_if_possible(proc, elems, dbg)
    _enrich_with_icons_yolo(proc, elems, dbg)
    unique, d2 = _dedupe_by_name_smart(elems)
    d2.update({k: v for k, v in dbg.items() if k.startswith("ocr_") or k.startswith("icon_yolo")})
    slim = _strip_fields_for_llm(unique)
    return {"screen": proc.get("screen"), "timestamp": proc.get("timestamp"), "elements": slim, "_debug": d2}

# -------------------- endpoints --------------------
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "controller": APP_NAME, "version": APP_VER}

@app.get("/config")
async def config():
    return {
        "status": "ok",
        "sut_state_url": SUT_STATE_URL,
        "sut_timeout_sec": SUT_TIMEOUT_SEC,
        "strip_fields": sorted(STRIP_FIELDS),
        "dedupe_mode": f"{DEDUPE_KEY_MODE}+ct_rank+canon",
        # OCR
        "ocr_enabled": OCR_ENABLED,
        "ocr_sync_budget": OCR_SYNC_BUDGET,
        "tesseract_lang": TESSERACT_LANG,
        "tessdata_prefix": TESSDATA_PREFIX,
        "ocr_queue_size": len(app.state.ocr_queue),
        "ocr_max_area_frac": OCR_MAX_AREA_FRAC,
        "ocr_min_char": OCR_MIN_CHAR,
        "ocr_max_char": OCR_MAX_CHAR,
        "ocr_min_conf_empty": OCR_MIN_CONF_EMPTY,
        "ocr_min_conf_dup": OCR_MIN_CONF_DUP,
        "ocr_min_conf_short": OCR_MIN_CONF_SHORT,
        "accept_lowconf_min": OCR_ACCEPT_LOWCONF_MIN,
        "rect_pad_px": RECT_PAD_PX,
        "use_hard_filters": OCR_USE_HARD_FILTERS,
        "thin_line_hard_px": THIN_LINE_HARD_PX,
        "thin_line_soft_w": THIN_LINE_SOFT_W,
        "thin_line_soft_h": THIN_LINE_SOFT_H,
        "try_invert": TRY_INVERT,
        "fallback_lang": ("" if OCR_FAST_MODE else FALLBACK_LANG),
        "ocr_fast_mode": OCR_FAST_MODE,
        "ocr_cooldown_sec": OCR_COOLDOWN_SEC,
        "ocr_parallel": _PAR,
        "ocr_char_whitelist": OCR_CHAR_WHITELIST,
        "ocr_strict_whitelist": OCR_STRICT_WHITELIST,
        # YOLO icons
        "icon_yolo_enabled": ICON_YOLO_ENABLED,
        "icon_yolo_available": _YOLO_AVAILABLE,
        "icon_yolo_model_path": ICON_YOLO_MODEL_PATH,
        "icon_yolo_model_loaded": _YOLO_MODEL is not None,
        "icon_yolo_conf": ICON_YOLO_CONF,
        "icon_yolo_iou": ICON_YOLO_IOU,
        "icon_match_min_iou": ICON_MATCH_MIN_IOU,
        "icon_only_on_empty": ICON_ONLY_ON_EMPTY,
        "icon_detect_whole_screen": ICON_DETECT_WHOLE_SCREEN,
        "icon_box_min_w": ICON_BOX_MIN_W,
        "icon_box_min_h": ICON_BOX_MIN_H,
        "icon_box_max_w": ICON_BOX_MAX_W,
        "icon_box_max_h": ICON_BOX_MAX_H,
        "icon_class_overrides": ICON_CLASS_OVERRIDES,
        "icon_name_semantics": ICON_NAME_SEMANTICS,
    }

@app.get("/debug/last-sut-error")
async def debug_last_sut_error():
    return {"status": "ok", "last": app.state.last_sut_error}

@app.get("/debug/ping-sut")
async def debug_ping_sut():
    try:
        async with httpx.AsyncClient(timeout=_httpx_timeout(), trust_env=False) as client:
            r = await client.post(SUT_STATE_URL, json={})
            ok = r.status_code
        return {"status": "ok", "http_status": ok}
    except Exception as e:
        app.state.last_sut_error = {"type": type(e).__name__, "url": SUT_STATE_URL, "message": str(e)}
        raise HTTPException(status_code=502, detail=f"ping failed: {app.state.last_sut_error.get('message','')}")

@app.post("/state/raw")
async def state_raw():
    return await _fetch_raw_state()

@app.post("/state/filtered")
async def state_filtered():
    raw = await _fetch_raw_state()
    return _filter_state_for_inspect(raw)

@app.post("/state/for-llm")
async def state_for_llm():
    raw = await _fetch_raw_state()
    return _filter_state_for_llm(raw)

@app.post("/state/fallback-full")
async def state_fallback_full():
    return app.state.last_raw_state or await _fetch_raw_state()

# ---- OCR queue debug ----
@app.get("/debug/ocr-queue")
async def debug_ocr_queue(limit: int = 50):
    data = app.state.ocr_queue[-limit:]
    return {"status": "ok", "count": len(data), "total": len(app.state.ocr_queue), "items": data}

@app.post("/debug/ocr-queue/clear")
async def debug_ocr_queue_clear():
    n = len(app.state.ocr_queue)
    app.state.ocr_queue.clear()
    return {"status": "ok", "cleared": n}

# ---- YOLO debug ----
@app.get("/debug/icon-yolo")
async def debug_icon_yolo():
    _load_yolo_once()
    return {
        "status": "ok",
        "available": _YOLO_AVAILABLE,
        "enabled": ICON_YOLO_ENABLED,
        "model_path": ICON_YOLO_MODEL_PATH,
        "loaded": _YOLO_MODEL is not None,
        "class_names": _YOLO_CLASS_NAMES,
        "icon_name_semantics": ICON_NAME_SEMANTICS,
    }

# ---- main ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("controller_service:app", host="0.0.0.0", port=int(os.getenv("CONTROLLER_PORT", "18800")), reload=True)
