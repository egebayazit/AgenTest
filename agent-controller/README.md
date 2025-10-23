🧩 Agent Controller – Quick Setup Guide
🚀 1. Ne İşe Yarar?

agent-controller, SUT’tan gelen ham UI durumunu ( /state ) işler:

RAW → FILTERED (OCR + YOLO) → FOR-LLM (final paket)


OCR (Tesseract): Adı boş öğelere metinden ad verir.

YOLO (Icon Detection): Simgeyi tanır (icon, text, container vs).

LLM Paketleme: Gereksiz alanları atar, sade JSON üretir.

⚙️ 2. Gereksinimler
Gereksinim	Açıklama
Python 3.11+	(Windows önerilir, Linux/macOS da olur)
Tesseract OCR	🔗 Kurulum

YOLO model (.pt)	🔗 İndir – orasul/deki-yolo

PIP paketleri	pip install -r requirements.txt

🧰 3. Kurulum
3.1 Python Ortamı
cd C:\Projects\AgenTest\agent-controller
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

3.2 Tesseract OCR

UB Mannheim Tesseract
’dan indir.
(kurulum yolu genelde C:\Program Files\Tesseract-OCR\tesseract.exe)

.env içine ekle:

OCR_ENABLED=True
TESSERACT_LANG=tur+eng
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
OCR_PARALLEL=8
OCR_SYNC_BUDGET=64

3.3 YOLO Modeli
pip install "huggingface_hub[cli]"
hf download orasul/deki-yolo --include "best.pt" --local-dir models\yolo
Rename-Item models\yolo\best.pt models\yolo\deki-best.pt -Force


.env içine ekle:

ICON_YOLO_ENABLED=True
ICON_YOLO_MODEL_PATH=models\yolo\deki-best.pt
ICON_YOLO_CONF=0.30
ICON_YOLO_IOU=0.50
ICON_MATCH_MIN_IOU=0.12
ICON_ONLY_ON_EMPTY=True
ICON_DETECT_WHOLE_SCREEN=True

3.4 SUT ve Servis Ayarları (.env)
SUT_STATE_URL=http://127.0.0.1:18080/state
SUT_TIMEOUT_SEC=45
HOST=0.0.0.0
PORT=18800
INCLUDE_DUPLICATES_FOR_LLM=True

▶️ 4. Servisi Başlatma
uvicorn controller_service:app --host 0.0.0.0 --port 18800 --reload


Loglarda INFO: Uvicorn running on http://0.0.0.0:18800 görüyorsan her şey hazır.

🧪 5. Test Etme (PowerShell)
# config ve YOLO durumu
Invoke-RestMethod http://127.0.0.1:18800/config
Invoke-RestMethod http://127.0.0.1:18800/debug/icon-yolo

# state testleri
$raw = Invoke-RestMethod -Method Post http://127.0.0.1:18800/state/raw -Body '{}'
$filtered = Invoke-RestMethod -Method Post http://127.0.0.1:18800/state/filtered -Body '{}'
$forllm = Invoke-RestMethod -Method Post http://127.0.0.1:18800/state/for-llm -Body '{}'

"Counts → raw:$($raw.elements.Count) filtered:$($filtered.elements.Count) for-llm:$($forllm.elements.Count)"


Beklenen:

icon_yolo_model_loaded=True

ocr_attempted > 0

detections_total > 0

for-llm içinde adlandırılmış elementler

🧩 6. Özet
Endpoint	Amaç
POST /state/raw	SUT’tan ham state
POST /state/filtered	OCR + YOLO + debug
POST /state/for-llm	Sade, LLM’e gönderilecek veri
GET /config	Aktif konfigürasyon
GET /debug/icon-yolo	YOLO yük durumu