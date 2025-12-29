from __future__ import annotations

import os
import re
import io
import base64
import time
import unicodedata
import hashlib
import html
import multiprocessing
import concurrent.futures as cf
from typing import Any, Dict, List, Tuple, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- Optional Deps ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from PIL import Image, ImageOps, ImageFilter
import pytesseract

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================

APP_NAME = "AgenTest Controller"
APP_VER = "0.22.0-MASTER-FIXED"

# Service URLs
SUT_STATE_URL = os.getenv("SUT_STATE_URL", "")
ODS_URL = os.getenv("ODS_URL", "http://127.0.0.1:8000/parse").strip()
SUT_TIMEOUT_SEC = int(os.getenv("SUT_TIMEOUT_SEC", "45"))
ODS_TIMEOUT_SEC = int(os.getenv("ODS_TIMEOUT_SEC", "30"))

# Deduplication Strategy
# .env ayarlarını zorla okuması için string kontrolü yapıyoruz
INCLUDE_DUPLICATES_FOR_LLM = os.getenv("INCLUDE_DUPLICATES_FOR_LLM", "0") in ("1", "true", "True")
DEDUPE_KEY_MODE = os.getenv("DEDUPE_KEY_MODE", "name").lower()

# OCR Config
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_ENABLED = os.getenv("OCR_ENABLED", "1") not in ("0", "false", "False", "")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
OCR_FAST_MODE = os.getenv("OCR_FAST_MODE", "1") in ("1", "true", "True")
OCR_SYNC_BUDGET = int(os.getenv("OCR_SYNC_BUDGET", "12"))
OCR_MAX_AREA_FRAC = float(os.getenv("OCR_MAX_AREA_FRAC", "0.95"))
OCR_MIN_CHAR = int(os.getenv("OCR_MIN_CHAR", "2"))
OCR_MAX_CHAR = int(os.getenv("OCR_MAX_CHAR", "200"))
OCR_MIN_CONF_EMPTY = int(os.getenv("OCR_MIN_CONF_EMPTY", "22"))
OCR_ACCEPT_LOWCONF_MIN = int(os.getenv("OCR_ACCEPT_LOWCONF_MIN", "12"))

RECT_PAD_PX = int(os.getenv("RECT_PAD_PX", "3"))
OCR_COOLDOWN_SEC = int(os.getenv("OCR_COOLDOWN_SEC", "6"))
OCR_CHAR_WHITELIST = os.getenv("OCR_CHAR_WHITELIST", "")
TRY_INVERT = os.getenv("TRY_INVERT", "0") in ("1", "true", "True")

# Parallelism
_PAR = int(os.getenv("OCR_PARALLEL", "0"))
if _PAR <= 0:
    try:
        _PAR = max(2, min(8, multiprocessing.cpu_count()))
    except Exception:
        _PAR = 4

# YOLO Config
ICON_YOLO_ENABLED = os.getenv("ICON_YOLO_ENABLED", "1") in ("1", "true", "True")
ICON_YOLO_MODEL_PATH = os.getenv("ICON_YOLO_MODEL_PATH", "").strip()
ICON_YOLO_CONF = float(os.getenv("ICON_YOLO_CONF", "0.35"))
ICON_YOLO_IOU = float(os.getenv("ICON_YOLO_IOU", "0.5"))
ICON_MATCH_MIN_IOU = float(os.getenv("ICON_MATCH_MIN_IOU", "0.12"))
ICON_ONLY_ON_EMPTY = os.getenv("ICON_ONLY_ON_EMPTY", "0") in ("1", "true", "True")
ICON_DETECT_WHOLE_SCREEN = os.getenv("ICON_DETECT_WHOLE_SCREEN", "1") in ("1", "true", "True")
ICON_BOX_MIN_W = int(os.getenv("ICON_BOX_MIN_W", "6"))
ICON_BOX_MIN_H = int(os.getenv("ICON_BOX_MIN_H", "6"))
ICON_BOX_MAX_W = int(os.getenv("ICON_BOX_MAX_W", "1600"))
ICON_BOX_MAX_H = int(os.getenv("ICON_BOX_MAX_H", "1600"))
ICON_CLASS_OVERRIDES = (os.getenv("ICON_CLASS_OVERRIDES") or "").strip()

ICON_NAME_SEMANTICS: Dict[str, str] = {
    "View": "container",
    "ImageView": "icon",
    "Text": "text",
    "Line": "separator",
}

STRIP_FIELDS = {"controlType", "enabled", "patterns", "idx", "hash"}

# ============================================================================
# 2. APP STATE & CACHES
# ============================================================================

app = FastAPI(title=APP_NAME, version=APP_VER)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.last_raw_state: Optional[Dict[str, Any]] = None
app.state.last_sut_error: Optional[Dict[str, Any]] = None
app.state.last_ods_error: Optional[Dict[str, Any]] = None
app.state.ocr_queue: List[Dict[str, Any]] = []

_OCR_CACHE: Dict[Tuple, Tuple[str, float]] = {}
_PREPROC_CACHE: Dict[Tuple, Image.Image] = {}
_OCR_CACHE_MAX = 800
_PREPROC_MAX = 400
_COOLDOWN: Dict[Tuple, float] = {}
_YOLO_MODEL: Optional[YOLO] = None
_YOLO_CLASS_NAMES: Dict[int, str] = {}


# ============================================================================
# 3. UTILITIES (Geometry, Text, HTTP)
# ============================================================================

def _httpx_timeout() -> httpx.Timeout:
    return httpx.Timeout(SUT_TIMEOUT_SEC, read=SUT_TIMEOUT_SEC, connect=5.0)

def _ods_timeout() -> httpx.Timeout:
    return httpx.Timeout(ODS_TIMEOUT_SEC, read=ODS_TIMEOUT_SEC, connect=5.0)

def _screen_size(raw: Dict[str, Any]) -> Tuple[int, int]:
    scr = raw.get("screen") or {}
    return int(scr.get("w", 0)), int(scr.get("h", 0))

def _rect_area(el: Dict[str, Any]) -> int:
    r = (el.get("rect") or {})
    w = max(0, int(r.get("r", 0)) - int(r.get("l", 0)))
    h = max(0, int(r.get("b", 0)) - int(r.get("t", 0)))
    return w * h

def _extract_rect_xywh(rect_obj: Any) -> Optional[Dict[str, float]]:
    if not isinstance(rect_obj, dict): return None
    if all(k in rect_obj for k in ("x", "y", "w", "h")):
        return {"x": float(rect_obj["x"]), "y": float(rect_obj["y"]), "w": float(rect_obj["w"]), "h": float(rect_obj["h"])}
    if all(k in rect_obj for k in ("l", "r", "t", "b")):
        return {"x": float(rect_obj["l"]), "y": float(rect_obj["t"]), "w": float(rect_obj["r"]) - float(rect_obj["l"]), "h": float(rect_obj["b"]) - float(rect_obj["t"])}
    return None

def _rect_from_element(el: Dict[str, Any], screen: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    rect = _extract_rect_xywh(el.get("rect"))
    if rect: return rect
    center = el.get("center")
    if isinstance(center, dict) and "x" in center and "y" in center:
        size = 12.0
        cx, cy = float(center["x"]), float(center["y"])
        return {"x": cx - size/2, "y": cy - size/2, "w": size, "h": size}
    return None

def _center_from_element(el: Dict[str, Any], rect: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    center = el.get("center")
    if isinstance(center, dict) and "x" in center and "y" in center:
        return {"x": float(center["x"]), "y": float(center["y"])}
    if rect:
        return {"x": rect["x"] + rect["w"]/2.0, "y": rect["y"] + rect["h"]/2.0}
    return None

def _extract_screenshot_b64(raw: Dict[str, Any]) -> Optional[str]:
    if not isinstance(raw, dict): return None
    sc = raw.get("screenshot")
    if isinstance(sc, dict):
        # Try to get b64 from screenshot object first
        result = sc.get("b64") or sc.get("image_b64") or sc.get("data")
        if result:
            return result
        # JVM state: screenshot object may only have "format", actual b64 is at root
    if isinstance(sc, str) and len(sc) > 1000:
        return sc
    return raw.get("screenshot_base64") or raw.get("b64")

def _strip_html_basic(s: str) -> str:
    if not s: return ""
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _post_normalize(s: str) -> str:
    if not s: return s
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _effective_name(el: Dict[str, Any]) -> str:
    raw = (el.get("name") or "").strip()
    return raw if raw else (el.get("name_ocr") or "").strip()

def _normalized_name_from_icon(el: Dict[str, Any]) -> Optional[str]:
    label = (el.get("name_icon") or "").strip()
    if not label: return None
    return ICON_NAME_SEMANTICS.get(label, label)

def _add_element_type_info(el: Dict[str, Any]) -> str:
    if "type" in el: return el["type"]
    ct = str(el.get("controlType", "")).lower()
    if ct in ("button", "checkbox", "radiobutton", "togglebutton"): return "button"
    if ct in ("edit", "textbox", "document"): return "input"
    if ct in ("text", "label", "statictext"): return "text"
    if ct in ("combobox", "list", "listitem", "treeitem"): return "list"
    if ct in ("image", "icon"): return "icon"
    if ct in ("window", "pane", "group"): return "container"
    return "unknown"

def _ct_rank(el: Dict[str, Any]) -> int:
    ct = (el.get("controlType") or "Other")
    prio = {"Text": 1, "Edit": 2, "Button": 3, "Window": 9, "Other": 10}
    return prio.get(ct, 10)

# ============================================================================
# 4. OCR & YOLO CORE LOGIC (FULL IMPLEMENTATION)
# ============================================================================

def _pad_and_crop(img: Image.Image, rect: Dict[str, Any]) -> Image.Image:
    img_w, img_h = int(img.width), int(img.height)
    try:
        l, t = int(rect.get("x", 0)), int(rect.get("y", 0))
        w, h = int(rect.get("w", 0)), int(rect.get("h", 0))
        r, b = l + w, t + h
    except: return img
    pad = RECT_PAD_PX
    l, t = max(0, l-pad), max(0, t-pad)
    r, b = min(img_w, r+pad), min(img_h, b+pad)
    if r <= l or b <= t: return img
    return img.crop((l, t, r, b))

def _ocr_try(crop: Image.Image, lang: str, psm: int) -> Tuple[str, float]:
    cfg = f"--psm {psm}"
    if OCR_CHAR_WHITELIST: cfg += f' -c tessedit_char_whitelist="{OCR_CHAR_WHITELIST}"'
    try:
        data = pytesseract.image_to_data(crop, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
        words = data.get("text", [])
        confs = data.get("conf", [])
        pairs = [(w.strip(), float(c)) for w, c in zip(words, confs) if w and w.strip() and float(c) >= 0]
        if not pairs: return "", 0.0
        text = " ".join(w for w, _ in pairs)
        avg_conf = sum(c for _, c in pairs) / len(pairs)
        return _post_normalize(text), avg_conf
    except: return "", 0.0

def _best_psm_candidates(rect: Dict[str, Any]) -> List[int]:
    h = max(1, int(rect.get("h", 0)))
    w = max(1, int(rect.get("w", 0)))
    return [7, 11] if (h <= 40 or w >= 4*h) else [6, 11, 3]

def _screenshot_from_raw_img(raw: Dict[str, Any]) -> Tuple[Optional[Image.Image], Optional[str]]:
    b64 = _extract_screenshot_b64(raw)
    if not b64: return None, None
    try:
        data = base64.b64decode(b64, validate=False)
        md5 = hashlib.md5(data).hexdigest()
        return Image.open(io.BytesIO(data)).convert("RGB"), md5
    except: return None, None

def _enrich_with_ocr_if_possible(raw: Dict[str, Any], elements: List[Dict[str, Any]], dbg: Dict) -> None:
    if not OCR_ENABLED: 
        dbg["ocr_skipped"] = "disabled"
        return
        
    img, shash = _screenshot_from_raw_img(raw)
    if not img: 
        dbg["ocr_skipped"] = "no_image"
        return

    candidates = []
    for el in elements:
        if not _effective_name(el):
            candidates.append(el)
    
    candidates.sort(key=lambda e: (_ct_rank(e), _rect_area(e)))
    workset = candidates[:OCR_SYNC_BUDGET]

    def _process_ocr(el):
        r = _rect_from_element(el, None)
        if not r: return {}
        # convert to simple rect for crop
        # _rect_from_element returns x,y,w,h
        
        psms = _best_psm_candidates(r)
        best_txt, best_conf = "", 0.0
        
        crop = _pad_and_crop(img, r)
        
        t, c = _ocr_try(crop, TESSERACT_LANG, psms[0])
        if c > best_conf: best_txt, best_conf = t, c
        
        if TRY_INVERT and best_conf < 50:
            t, c = _ocr_try(ImageOps.invert(ImageOps.grayscale(crop)), TESSERACT_LANG, psms[0])
            if c > best_conf: best_txt, best_conf = t, c
            
        if best_conf >= OCR_ACCEPT_LOWCONF_MIN and len(best_txt) >= OCR_MIN_CHAR:
            return {"name_ocr": best_txt, "name_ocr_conf": best_conf}
        return {}

    if workset:
        with cf.ThreadPoolExecutor(max_workers=_PAR) as ex:
            results = list(ex.map(_process_ocr, workset))
        
        matched = 0
        for el, res in zip(workset, results):
            if res:
                el.update(res)
                matched += 1
        
        if dbg is not None:
            dbg["ocr_attempted"] = len(workset)
            dbg["ocr_matched"] = matched

def _yolo_is_ready() -> bool:
    return _YOLO_AVAILABLE and bool(ICON_YOLO_MODEL_PATH)

def _load_yolo_once() -> None:
    global _YOLO_MODEL, _YOLO_CLASS_NAMES
    if _YOLO_MODEL is not None: return
    if not _yolo_is_ready(): return
    try:
        _YOLO_MODEL = YOLO(ICON_YOLO_MODEL_PATH)
        names = getattr(_YOLO_MODEL.model, "names", {})
        _YOLO_CLASS_NAMES = {int(k): str(v) for k, v in names.items()}
    except: 
        _YOLO_MODEL = None

def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def _enrich_with_icons_yolo(raw: Dict[str, Any], elements: List[Dict[str, Any]], dbg: Dict) -> None:
    if not ICON_YOLO_ENABLED or not _yolo_is_ready(): 
        dbg["yolo_skipped"] = "disabled_or_not_ready"
        return
    
    _load_yolo_once()
    if _YOLO_MODEL is None: return

    img, _ = _screenshot_from_raw_img(raw)
    if not img: 
        dbg["yolo_skipped"] = "no_image"
        return

    try:
        results = _YOLO_MODEL.predict(img, conf=ICON_YOLO_CONF, iou=ICON_YOLO_IOU, verbose=False)
        if not results: return
        
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = _YOLO_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            if w < ICON_BOX_MIN_W or h < ICON_BOX_MIN_H: continue
            detections.append({"box": xyxy, "label": cls_name, "conf": conf})

        matched = 0
        for el in elements:
            if not ICON_ONLY_ON_EMPTY or not _effective_name(el):
                r = _rect_from_element(el, None)
                if not r: continue
                el_box = [r["x"], r["y"], r["x"]+r["w"], r["y"]+r["h"]]
                
                best_iou = 0
                best_det = None
                for det in detections:
                    iou_val = _iou(el_box, det["box"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_det = det
                
                if best_det and best_iou >= ICON_MATCH_MIN_IOU:
                    el["name_icon"] = best_det["label"]
                    el["name_icon_conf"] = best_det["conf"]
                    matched += 1
                    
        if dbg is not None:
            dbg["yolo_detections"] = len(detections)
            dbg["yolo_matched"] = matched

    except Exception as e:
        print(f"YOLO Error: {e}")

# ============================================================================
# 5. JVM STATE HANDLING (Does NOT affect WinDriver/Windows)
# ============================================================================

def _is_jvm_state(raw: Dict[str, Any]) -> bool:
    """Check if state is from JVM agent (has stateType='jvm' or nested children)"""
    if raw.get("stateType") == "jvm":
        return True
    # Fallback: check if elements have 'children' or 'class' (JVM indicators)
    elements = raw.get("elements", [])
    if elements and isinstance(elements, list) and len(elements) > 0:
        first_el = elements[0]
        if isinstance(first_el, dict) and ("children" in first_el or "class" in first_el):
            return True
    return False

def _flatten_jvm_elements(elements: List[Dict[str, Any]], result: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Recursively flatten nested JVM elements with children"""
    if result is None:
        result = []
    
    for el in elements:
        if not isinstance(el, dict):
            continue
        
        # Add current element (will be processed later)
        result.append(el)
        
        # Recursively process children
        children = el.get("children", [])
        if children and isinstance(children, list):
            _flatten_jvm_elements(children, result)
    
    return result

def _normalize_jvm_element(el: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert JVM element format to standard format with center/name/type"""
    # Extract name: JVM uses 'text' field
    name = (el.get("text") or "").strip()
    if not name:
        return None
    
    # Calculate center from x, y, width, height
    try:
        x = float(el.get("x", 0))
        y = float(el.get("y", 0))
        w = float(el.get("width", 0))
        h = float(el.get("height", 0))
        
        # Skip elements without valid dimensions
        if w <= 0 or h <= 0:
            return None
        
        cx = x + w / 2.0
        cy = y + h / 2.0
    except (ValueError, TypeError):
        return None
    
    # Determine type from JVM class
    jvm_class = (el.get("class") or "").lower()
    el_type = "unknown"
    if "button" in jvm_class:
        el_type = "button"
    elif "text" in jvm_class or "label" in jvm_class:
        el_type = "text"
    elif "edit" in jvm_class or "input" in jvm_class or "field" in jvm_class:
        el_type = "input"
    elif "combo" in jvm_class or "list" in jvm_class:
        el_type = "list"
    elif "menu" in jvm_class:
        el_type = "menu"
    elif "panel" in jvm_class or "pane" in jvm_class or "frame" in jvm_class:
        el_type = "container"
    
    return {
        "name": name,
        "center": {"x": cx, "y": cy},
        "type": el_type
    }

def _process_jvm_state(raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process JVM state: flatten and normalize elements"""
    elements = raw.get("elements", [])
    screen = raw.get("screen") or {}
    
    # Flatten nested children
    flat_elements = _flatten_jvm_elements(elements)
    
    # Normalize each element
    normalized = []
    for el in flat_elements:
        norm_el = _normalize_jvm_element(el)
        if norm_el:
            normalized.append(norm_el)
    
    return normalized, screen

# ============================================================================
# 6. EXTERNAL SERVICES (Fetch State)
# ============================================================================

async def _fetch_raw_state() -> Dict[str, Any]:
    """Fetch raw state from SUT"""
    try:
        async with httpx.AsyncClient(timeout=_httpx_timeout(), trust_env=False) as client:
            r = await client.post(SUT_STATE_URL, json={})
            r.raise_for_status()
            data = r.json()
            app.state.last_raw_state = data
            app.state.last_raw_ts = time.time()
            app.state.last_sut_error = None
            return data
    except Exception as e:
        app.state.last_sut_error = {"msg": str(e), "url": SUT_STATE_URL}
        raise HTTPException(502, f"SUT Error: {e}")

async def _call_ods_with_base64(b64: str) -> Dict[str, Any]:
    """Send base64 screenshot to ODS"""
    if not ODS_URL: return {}
    try:
        t_start = time.time()
        print(f"[ODS] Starting request... b64_size={len(b64)} bytes")
        
        async with httpx.AsyncClient(timeout=_ods_timeout(), trust_env=False, follow_redirects=True) as client:
            t_before_post = time.time()
            r = await client.post(ODS_URL, json={"base64_image": b64})
            t_after_post = time.time()
            r.raise_for_status()
            
            result = r.json()
            t_end = time.time()
            
            print(f"[ODS] Complete in {t_end - t_start:.2f}s: network={t_after_post - t_before_post:.2f}s, parse={t_end - t_after_post:.2f}s")
            return result
    except Exception as e:
        app.state.last_ods_error = {"msg": str(e), "url": ODS_URL}
        print(f"[ODS] Error after {time.time() - t_start:.2f}s: {e}")
        raise

def _elements_from_ods_response(ods_raw: dict, screen: dict) -> list[dict]:
    """Convert ODS response to element list with high precision coordinate conversion"""
    # Ekran boyutlarını float olarak alalım
    w = float(screen.get("w", 1920))
    h = float(screen.get("h", 1080))
    elements = []
    
    for item in ods_raw.get("parsed_content_list", []):
        if item.get("type") not in ("text", "icon"): continue
        bbox = item.get("bbox", [])
        if len(bbox) != 4: continue
        
        # ODS/OmniParser genellikle [y1, x1, y2, x2] veya [x1, y1, x2, y2] gönderir.
        # Paylaştığın loglara göre format: [x1, y1, x2, y2] ve 0-1 arası float.
        x1, y1, x2, y2 = bbox
        
        # HATA DÜZELTME: Eğer değerler 0-1 arasındaysa doğrudan ekran boyutuyla çarpılır.
        # Eğer değerler 0-1000 arasındaysa önce 1000'e bölünmelidir.
        # OmniParser loguna göre (0.39...) değerler 0-1 arasındadır.
        
        cx = ((x1 + x2) / 2.0) * w
        cy = ((y1 + y2) / 2.0) * h
        
        elements.append({
            "name": item.get("content") or "",
            "type": item.get("type"),
            "center": {"x": round(cx, 1), "y": round(cy, 1)}
        })
    return elements

# ============================================================================
# 6. FORMATTING & FINAL LOGIC
# ============================================================================

def _format_compact_text_for_llm(elements: List[Dict[str, Any]]) -> str:
    """
    LLM Formatı: (x,y) [Type] Name | Value: val | ID: aid
    """
    lines = ["Format: (x,y) [Type] Name | Value | AutomationID"]
    lines.append("-" * 60)
    
    for el in elements:
        c = el.get("center", {})
        cx, cy = int(c.get("x", 0)), int(c.get("y", 0))
        etype = el.get("type", "ui")
        
        name = (el.get("name") or "").strip()
        val = (el.get("value") or "").strip()
        aid = (el.get("automationId") or "").strip()

        # Bilgileri parça parça oluştur
        parts = [f"[{etype}] {name if name else 'NoName'}"]
        if val: 
            # Değer çok uzunsa (örn uzun URL) LLM'i yormamak için kısaltıyoruz
            short_val = (val[:47] + "...") if len(val) > 50 else val
            parts.append(f"Value: {short_val}")
        if aid: 
            parts.append(f"ID: {aid}")
        
        content = " | ".join(parts)
        line = f"({cx},{cy}) {content}"
        lines.append(line)
        
    return "\n".join(lines)

def _dedupe_by_name_smart(elements: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict]:
    """
    Dedupe Mantığı:
    1. Sadece (Name, X, Y) üçlüsü birebir aynıysa eleme yap.
    2. Eğer aynı yerdeyse; name, automationId ve value alanlarından en dolu olanı tut.
    3. Farklı koordinattaki aynı isimli elementlere dokunma (ikisi de kalsın).
    """
    unique_map = {}
    
    for el in elements:
        # Verileri temizle
        name = (el.get("name") or "").strip()
        val = (el.get("value") or "").strip()
        aid = (el.get("automationId") or "").strip()
        
        # Koordinatları al
        c = el.get("center", {})
        try:
            cx, cy = int(c.get("x", 0)), int(c.get("y", 0))
        except:
            continue
        
        # KEY: İsim + Koordinat (Aynı isimli ama farklı yerdeki elementler farklı key alır)
        unique_key = (name, cx, cy)
        
        if unique_key not in unique_map:
            # Henüz bu isimde ve bu koordinatta bir element yok, ekle.
            unique_map[unique_key] = el
        else:
            # AYNI YERDE AYNI İSİMLİ BİR ELEMENT VAR!
            existing_el = unique_map[unique_key]
            
            # Doluluk puanı hesapla (Hangisinde daha çok bilgi var?)
            # Name zaten key içinde olduğu için puanlamada Value ve ID kritik.
            def get_score(e):
                score = 0
                if (e.get("name") or "").strip(): score += 1
                if (e.get("automationId") or "").strip(): score += 1
                if (e.get("value") or "").strip(): score += 2  # Value genelde en ayırt edici veridir (URL vb.)
                return score

            new_score = get_score(el)
            old_score = get_score(existing_el)
            
            # Eğer yeni gelen daha doluysa eskisinin yerine yaz
            if new_score > old_score:
                unique_map[unique_key] = el
            # Puanlar eşitse ControlType önceliğine bak (opsiyonel, Hyperlink genelde daha iyidir)
            elif new_score == old_score:
                if str(el.get("controlType")).lower() == "hyperlink":
                    unique_map[unique_key] = el
                
    result = list(unique_map.values())
    return result, {"dedupe_count": len(elements) - len(result)}

def _finalize_elements_for_llm(elements: List[Dict[str, Any]], screen: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    temp_elements = []
    # Blacklist'i sadece koordinat hatası için bıraktık
    BLACKLIST = {"0,0"} 
    
    for el in elements or []:
        # Ham verileri topla
        raw_name = (el.get("name") or "").strip()
        val = (el.get("value") or "").strip()
        aid = (el.get("automationId") or "").strip()
        
        # Name boşsa OCR veya Icon denemesini buraya sakla
        name = raw_name or (el.get("name_ocr") or "").strip() or _normalized_name_from_icon(el) or ""

        # KURAL: En az bir tanesi dolu mu?
        if not any([name, val, aid]):
            continue
            
        # Blacklist kontrolü
        if any(x.lower() in BLACKLIST for x in [name, val, aid]):
            continue
        
        rect = _rect_from_element(el, screen)
        center = _center_from_element(el, rect)
        if not center: 
            continue
        
        # LLM element objesini oluştur
        temp_el = {
            "name": name,
            "value": val,
            "automationId": aid,
            "type": _add_element_type_info(el),
            "center": center
        }
        temp_elements.append(temp_el)
    
    # JSON yapısını bozmadan listeyi ata
    structured_list = temp_elements 

    # LLM metin görünümünü oluştur
    compact_view = _format_compact_text_for_llm(structured_list)

    return {
        "elements": structured_list,
        "llm_view": compact_view,
        "screen": screen,
        "count": len(structured_list)
    }

# ============================================================================
# 7. API ENDPOINTS
# ============================================================================

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "controller": APP_NAME, "version": APP_VER}

@app.post("/state/raw")
async def state_raw():
    return await _fetch_raw_state()

@app.post("/state/for-llm")
async def state_for_llm():
    """Standard SUT path (WinDriver/JVM) with Enrichment & Dedupe"""
    raw = await _fetch_raw_state()
    dbg = {}
    
    # === JVM STATE HANDLING (separate path, does not affect WinDriver) ===
    if _is_jvm_state(raw):
        dbg["state_type"] = "jvm"
        elems, screen = _process_jvm_state(raw)
        
        # Dedupe for JVM
        if not INCLUDE_DUPLICATES_FOR_LLM:
            elems, d = _dedupe_by_name_smart(elems)
            dbg.update(d)
        
        res = _finalize_elements_for_llm(elems, screen)
        res["_debug"] = dbg
        return res
    
    # === WINDRIVER/WINDOWS PATH (unchanged) ===
    dbg["state_type"] = "windriver"
    proc = raw
    elems = proc.get("elements") or []
    screen = proc.get("screen") or {}
    
    # Enrichment
    _enrich_with_ocr_if_possible(proc, elems, dbg)
    _enrich_with_icons_yolo(proc, elems, dbg)
    
    # Dedupe
    if not INCLUDE_DUPLICATES_FOR_LLM:
        elems, d = _dedupe_by_name_smart(elems)
        dbg.update(d)
        
    res = _finalize_elements_for_llm(elems, screen)
    res["_debug"] = dbg
    return res

@app.post("/state/from-ods")
async def state_from_ods():
    """ODS path (Visual Model) with Error Handling & Dedupe"""
    t_endpoint_start = time.time()
    print(f"[/state/from-ods] Request started")
    
    t_fetch_start = time.time()
    raw = await _fetch_raw_state()
    print(f"[/state/from-ods] fetch_raw_state took {time.time() - t_fetch_start:.2f}s")
    
    if not ODS_URL:
        raise HTTPException(501, "ODS_URL not set")

    b64 = _extract_screenshot_b64(raw)
    if not b64:
        raise HTTPException(400, "No screenshot in SUT state")

    # --- GENEL GEÇER ÇÖZÜM: RESMİN GERÇEK BOYUTLARINI AL ---
    img_bytes = base64.b64decode(b64)
    with Image.open(io.BytesIO(img_bytes)) as img:
        actual_w, actual_h = img.size
    actual_screen = {"w": actual_w, "h": actual_h}
    # -----------------------------------------------------

    try:
        t_ods_start = time.time()
        ods_data = await _call_ods_with_base64(b64)
        print(f"[/state/from-ods] ODS call took {time.time() - t_ods_start:.2f}s")
        
        if not ods_data or not ods_data.get("parsed_content_list"):
             raise HTTPException(502, "ODS returned empty data")
            
        # Hesaplamayı resmin gerçek boyutlarına (actual_screen) göre yapıyoruz
        ods_elements = _elements_from_ods_response(ods_data, actual_screen)
        
        # Koordinat yuvarlama (Token tasarrufu ve hassasiyet dengesi)
        for el in ods_elements:
            if "center" in el:
                el["center"]["x"] = int(el["center"]["x"])
                el["center"]["y"] = int(el["center"]["y"])

        if not INCLUDE_DUPLICATES_FOR_LLM:
             ods_elements, _ = _dedupe_by_name_smart(ods_elements)

        t_finalize_start = time.time()
        # Finalize ederken de gerçek boyutları kullanıyoruz
        result = _finalize_elements_for_llm(ods_elements, actual_screen)
        
        print(f"[/state/from-ods] finalize took {time.time() - t_finalize_start:.2f}s")
        print(f"[/state/from-ods] TOTAL TIME: {time.time() - t_endpoint_start:.2f}s")
        return result
        
    except Exception as e:
        print(f"[/state/from-ods] ODS Critical Error after {time.time() - t_endpoint_start:.2f}s: {e}")
        raise HTTPException(502, f"ODS Service Failed: {str(e)}")

@app.post("/debug/ods")
async def debug_ods_overlay():
    """
    Debug endpoint to fetch state from SUT, send to ODS, 
    save the overlay image locally, and return the ID mapping.
    """
    # 1. Fetch raw state from SUT to get the screenshot
    try:
        raw = await _fetch_raw_state()
        b64 = _extract_screenshot_b64(raw)
        if not b64:
            raise HTTPException(400, "No screenshot found in SUT state")
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch state from SUT: {e}")

    # 2. Call ODS service
    if not ODS_URL:
        raise HTTPException(501, "ODS_URL is not configured")
    
    try:
        start_time = time.time()
        ods_data = await _call_ods_with_base64(b64)
        latency = time.time() - start_time
    except Exception as e:
        raise HTTPException(502, f"ODS Service Failed: {e}")

    if not ods_data:
        raise HTTPException(502, "ODS returned empty response")

    # 3. Save Overlay Image (if available)
    overlay_b64 = ods_data.get("som_image_base64")
    saved_filename = None
    
    if overlay_b64:
        try:
            # Clean up base64 string if it has a header
            if "," in overlay_b64:
                overlay_b64 = overlay_b64.split(",")[1]
            
            img_data = base64.b64decode(overlay_b64)
            
            # Save to current directory with timestamp
            ts = int(time.time())
            filename = f"ods_debug_{ts}.png"
            with open(filename, "wb") as f:
                f.write(img_data)
            
            saved_filename = os.path.abspath(filename)
            print(f"🖼️  ODS Overlay saved to: {saved_filename}")
        except Exception as e:
            print(f"⚠️ Failed to save overlay image: {e}")

    # 4. Process and Return ID Mapping
    # OmniParser returns a list where the index usually corresponds to the ID in the overlay.
    parsed_content = ods_data.get("parsed_content_list", [])
    
    id_mapping = []
    for idx, item in enumerate(parsed_content):
        id_mapping.append({
            "id": idx,  # The overlay usually uses 0-based or 1-based indexing matching this order
            "type": item.get("type"),
            "content": item.get("content"),
            "bbox": item.get("bbox"),
            "source": item.get("source")
        })

    return {
        "status": "ok",
        "ods_latency": latency,
        "overlay_image_path": saved_filename,
        "element_count": len(id_mapping),
        "elements": id_mapping
    }


@app.get("/config")
async def config():
    return {
        "status": "ok",
        "sut_url": SUT_STATE_URL,
        "ods_url": ODS_URL,
        "ocr_enabled": OCR_ENABLED,
        "yolo_enabled": ICON_YOLO_ENABLED,
        "dedupe_enabled": not INCLUDE_DUPLICATES_FOR_LLM
    }

@app.post("/state/filtered")
async def state_filtered():
    raw = await _fetch_raw_state()
    return raw

if __name__ == "__main__":
    port = int(os.getenv("CONTROLLER_PORT", "18800"))
    uvicorn.run("controller_service:app", host="0.0.0.0", port=port, reload=True)