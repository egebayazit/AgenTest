# --- flash_attn safe stub (before any HF imports) ---
import os

# OMNI_NO_FLASH_STUB=1 ise bu blok hiçbir şey yapmaz
if os.environ.get("OMNI_NO_FLASH_STUB") != "1":
    try:
        import flash_attn  # noqa: F401
    except Exception:
        import sys, types
        if "flash_attn" not in sys.modules:
            sys.modules["flash_attn"] = types.ModuleType("flash_attn")
# ----------------------------------------------------

import sys
os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "sdpa")

import time
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
import uvicorn
from difflib import SequenceMatcher

# === Yeni: kök klasörü güvenli ekleme (Windows göreli yol sorunlarını azaltır)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# === Yeni: torch ile cihaz otomatik seçimi
import torch

from util.omniparser import Omniparser


def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')

    # === Değişti: YOLO ağı için varsayılanı 'yolov8n.pt' yap, mutlaklaşsın
    parser.add_argument('--som_model_path', type=str, default='yolov8n.pt',
                        help='Path to the YOLO/SoM model (e.g., yolov8n.pt or a custom .pt)')

    # === Florence2 tarafı
    parser.add_argument('--caption_model_name', type=str, default='florence2',
                        help='Caption model id/key (kept for compatibility)')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence',
                        help='Path or HF id for the caption model weights')

    # === Değişti: cihaz otomatik (cuda varsa cuda, değilse cpu)
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device,
                        help='Device to run the model: cuda | cpu')

    # Not: orijinalde BOX_TRESHOLD yazımı vardı; sistemde öyle bekleniyorsa dokunmayalım
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.01, 
                        help='Threshold for box detection (Lowered for v2.0)')

    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    args = parser.parse_args()
    return args


args = parse_arguments()

# === Yeni: Florence2'yi flash_attn’siz çalıştırmak için SDPA'ya zorla
os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "sdpa")

# === Yeni: som_model_path mutlak yola çevir
som_path = args.som_model_path
if not os.path.isabs(som_path):
    som_path = os.path.abspath(os.path.join(root_dir, som_path))

# === Config’i dict olarak hazırla
config = {
    'som_model_path': som_path, 
    'caption_model_name': args.caption_model_name,
    'caption_model_path': args.caption_model_path,
    'device': args.device,
    'BOX_TRESHOLD': args.BOX_TRESHOLD,
}

app = FastAPI()

# === Yeni: init’te hatayı okunur şekilde yakala
try:
    omniparser = Omniparser(config)
except Exception as e:
    import traceback
    traceback.print_exc()
    raise

# --- Request Modelleri ---
class ParseRequest(BaseModel):
    base64_image: str

class PointRequest(BaseModel):
    base64_image: str
    x: int
    y: int

class ContentIdRequest(BaseModel):
    base64_image: str
    content_id: str

# --- Yardımcı Fonksiyonlar ---
def get_image_size(base64_string):
    """Base64 string'den resim boyutlarını döner (width, height)"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image.size

# --- Endpointler ---

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    # dino_labled_img: Base64 string (işlenmiş resim)
    # parsed_content_list: [{'bbox': [0.1, 0.1, 0.2, 0.2], 'content': 'Save', ...}] (Normalize BBox)
    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

@app.post("/get-id-from-ods")
async def get_id_from_ods(request: PointRequest):
    """
    Verilen (x, y) koordinatındaki elementin ismini (content) döner.
    Çift monitör uyumludur (Base64 resmin boyutuna göre normalize eder).
    """
    try:
        # 1. Resim boyutlarını al (Normalize edebilmek için)
        width, height = get_image_size(request.base64_image)
        
        # 2. Gelen piksel koordinatlarını 0.0-1.0 arasına normalize et
        target_x_ratio = request.x / width
        target_y_ratio = request.y / height

        # 3. ODS'yi çalıştır
        print(f"Searching element at ({request.x}, {request.y}) [Ratio: {target_x_ratio:.4f}, {target_y_ratio:.4f}]")
        _, parsed_content_list = omniparser.parse(request.base64_image)

        found_element = None
        min_area = float('inf')

        # 4. Elementleri tara
        for el in parsed_content_list:
            # ODS bbox formatı normalize edilmiş [x1, y1, x2, y2] şeklindedir (util/utils.py analizine göre)
            bbox = el.get('bbox')
            if not bbox: continue

            x1, y1, x2, y2 = bbox

            # Nokta kutunun içinde mi?
            if x1 <= target_x_ratio <= x2 and y1 <= target_y_ratio <= y2:
                # En spesifik (en küçük) elementi seçmek için alan hesabı
                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h
                
                # Sadece 'content' alanı dolu olanları tercih et (veya type'ı icon/text olanları)
                content = el.get('content')
                if content and area < min_area:
                    min_area = area
                    found_element = content

        if found_element:
            return {"status": "success", "content_id": found_element}
        else:
            return {"status": "not_found", "message": f"No element found at ({request.x}, {request.y})"}

    except Exception as e:
        print(f"Error in get-id-from-ods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-coords-from-ods")
async def get_coords_from_ods(request: ContentIdRequest):
    """
    Verilen 'content_id' (element ismi) ile eşleşen elementin merkez koordinatlarını döner.
    """
    try:
        # 1. Resim boyutlarını al (Piksel koordinatına çevirmek için)
        width, height = get_image_size(request.base64_image)
        
        # 2. ODS'yi çalıştır
        print(f"Searching coordinates for: '{request.content_id}'")
        _, parsed_content_list = omniparser.parse(request.base64_image)

        target_name = request.content_id.lower().strip()
        best_match = None
        highest_ratio = 0.0

        # 3. Elementleri tara (Fuzzy Match ile)
        for el in parsed_content_list:
            content = el.get('content')
            if not content: continue
            
            el_name = str(content).lower().strip()
            
            # Tam eşleşme (Hız için önce buna bak)
            if el_name == target_name:
                best_match = el
                break
            
            # Benzerlik kontrolü (%80 üzeri)
            ratio = SequenceMatcher(None, target_name, el_name).ratio()
            if ratio > highest_ratio and ratio > 0.8:
                highest_ratio = ratio
                best_match = el

        if best_match:
            bbox = best_match.get('bbox') # [x1, y1, x2, y2] (Normalize)
            
            # Merkez noktasını hesapla (Ratio cinsinden)
            center_x_ratio = (bbox[0] + bbox[2]) / 2
            center_y_ratio = (bbox[1] + bbox[3]) / 2
            
            # Piksel koordinatına çevir
            pixel_x = int(center_x_ratio * width)
            pixel_y = int(center_y_ratio * height)
            
            return {
                "status": "success", 
                "x": pixel_x, 
                "y": pixel_y,
                "matched_name": best_match.get('content')
            }
        else:
            return {"status": "not_found", "message": f"Element '{request.content_id}' not found"}

    except Exception as e:
        print(f"Error in get-coords-from-ods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omnitool.omniparserserver.omniparserserver:app",
                host=args.host, port=args.port, reload=True)