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

    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    args = parser.parse_args()
    return args


args = parse_arguments()

# === Yeni: Florence2'yi flash_attn’siz çalıştırmak için SDPA'ya zorla
# (flash_attn Windows’ta sorun çıkarıyor diye global env ile ez.
#  transformers >=4.41'de bu bayrak SDPA/FlashAttention davranışını belirliyor.)
os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "sdpa")

# === Yeni: som_model_path mutlak yola çevir + fallback mesajları utils tarafında zaten var
# (get_yolo_model utils’inde fallback yazdıysak buna gerek yok;
#  ama yine de temiz olsun diye path’i normalize edelim)
som_path = args.som_model_path
if not os.path.isabs(som_path):
    som_path = os.path.abspath(os.path.join(root_dir, som_path))

# === Config’i dict olarak hazırla
config = {
    'som_model_path': som_path,  # utils.get_yolo_model içinde yoksa YOLO('yolov8n.pt') fallback yapar
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
    # Import-time hataları eziyetli olabiliyor; logla ve yeniden fırlat
    import traceback
    traceback.print_exc()
    raise

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

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

@app.post("/get-coords-from-ods")
async def get_coords_from_ods(request: ContentIdRequest):
    """
    Verilen 'content_id' (element ismi) ile eşleşen TÜM elementlerin
    koordinatlarını ve ID'lerini liste olarak döner.
    """
    try:
        # 1. Resim boyutlarını al
        width, height = get_image_size(request.base64_image)
        
        # 2. ODS'yi çalıştır
        print(f"Searching all coordinates for: '{request.content_id}'")
        _, parsed_content_list = omniparser.parse(request.base64_image)

        target_name = request.content_id.lower().strip()
        found_matches = []

        # 3. Elementleri tara (enumerate ile ID'yi de alıyoruz)
        for i, el in enumerate(parsed_content_list):
            content = el.get('content')
            if not content: continue
            
            el_name = str(content).lower().strip()
            
            # Eşleşme Kontrolü
            # 1. Tam Eşleşme
            is_exact_match = (el_name == target_name)
            
            # 2. Benzerlik Kontrolü (%80 üzeri)
            ratio = SequenceMatcher(None, target_name, el_name).ratio()
            is_fuzzy_match = (ratio > 0.8)

            # Eğer tam eşleşme veya yüksek benzerlik varsa listeye ekle
            if is_exact_match or is_fuzzy_match:
                bbox = el.get('bbox') # [x1, y1, x2, y2]
                
                # Merkez noktasını hesapla
                center_x_ratio = (bbox[0] + bbox[2]) / 2
                center_y_ratio = (bbox[1] + bbox[3]) / 2
                
                # Piksel koordinatına çevir
                pixel_x = int(center_x_ratio * width)
                pixel_y = int(center_y_ratio * height)
                
                found_matches.append({
                    "id": i,                # Elementin benzersiz ID'si (Index)
                    "x": pixel_x,           # Tıklanacak X
                    "y": pixel_y,           # Tıklanacak Y
                    "content": content,     # Orijinal metin
                    "match_score": 1.0 if is_exact_match else ratio # Ne kadar benziyor?
                })

        # 4. Sonuçları döndür
        if found_matches:
            # Sonuçları skora göre sıralayalım (En çok benzeyen en üstte)
            found_matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            return {
                "status": "success", 
                "count": len(found_matches),
                "matches": found_matches
            }
        else:
            return {
                "status": "not_found", 
                "message": f"No elements found matching '{request.content_id}'"
            }

    except Exception as e:
        print(f"Error in get-coords-from-ods: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/get-id-from-ods")
async def get_id_from_ods(request: PointRequest):
    """
    Verilen (x, y) koordinatındaki elementin ismini ve debug endpoint'teki ile
    aynı olan ID'sini (index numarasını) döner.
    """
    try:
        # 1. Resim boyutlarını al
        width, height = get_image_size(request.base64_image)
        
        # 2. Koordinatları normalize et (0.0 - 1.0 arasına çek)
        target_x_ratio = request.x / width
        target_y_ratio = request.y / height

        print(f"Searching element at ({request.x}, {request.y}) [Ratio: {target_x_ratio:.4f}, {target_y_ratio:.4f}]")
        
        # 3. ODS'yi çalıştır
        # Not: Eğer projenizde _call_ods_with_base64 fonksiyonu standart ise onu da kullanabilirsiniz.
        # Burada önceki kodunuzdaki omniparser.parse'ı kullandım, ikisi de aynı listeyi döndürmeli.
        _, parsed_content_list = omniparser.parse(request.base64_image)

        found_content = None
        found_id = None
        min_area = float('inf')

        # 4. Elementleri tara
        # BURASI ÖNEMLİ: 'enumerate' kullanarak debug endpointindeki gibi 
        # listenin sıra numarasını (i) alıyoruz. Bu bizim ID'miz oluyor.
        for i, el in enumerate(parsed_content_list):
            bbox = el.get('bbox')
            if not bbox: continue

            x1, y1, x2, y2 = bbox

            # Koordinat kutunun içinde mi?
            if x1 <= target_x_ratio <= x2 and y1 <= target_y_ratio <= y2:
                
                # Daha spesifik (küçük) elementi bulmak için alan hesabı
                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h
                
                content = el.get('content')
                
                # Eğer birden fazla kutu üst üsteyse (örn: ekranın tamamı ve içindeki buton),
                # en küçük alana sahip olanı (butonu) seçiyoruz.
                if content and area < min_area:
                    min_area = area
                    found_content = content
                    found_id = i  # Debug endpointindeki logic: ID = Index

        if found_content:
            return {
                "status": "success", 
                "content_name": found_content, 
                "element_id": found_id  # Artık null gelmeyecek, sayı gelecek (0, 1, 5 vs.)
            }
        else:
            return {
                "status": "not_found", 
                "message": f"No element found at ({request.x}, {request.y})"
            }

    except Exception as e:
        print(f"Error in get-id-from-ods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omnitool.omniparserserver.omniparserserver:app",
                host=args.host, port=args.port, reload=True)
    