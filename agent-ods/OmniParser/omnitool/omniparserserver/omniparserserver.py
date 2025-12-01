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
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import uvicorn

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

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omnitool.omniparserserver.omniparserserver:app",
                host=args.host, port=args.port, reload=True)
