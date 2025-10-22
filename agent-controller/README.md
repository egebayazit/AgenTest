README — Agent Controller
1) Bu modül ne yapar?

agent-controller, SUT’tan gelen ham UI durumunu (/state) alır ve aşağıdaki pipeline’ı çalıştırır:

Filtreleme & Dedupe: UI elementlerinde gereksiz/gürültülü/tekrarlı kayıtları ayıklar.

OCR (Tesseract): Adı boş veya dup olan öğeleri metinden okumaya çalışır.

İkon Tespiti (YOLO): OCR’dan sonra hâlâ adı boş kalan öğeler için, ekrandaki ikon/simgeleri YOLO ile etiketler (örn. icon, text, container, separator).

LLM Paketi: LLM’e gidecek sade ve yararlı alanları içeren state/for-llm çıktısını üretir.

Özetle: raw → filtered (OCR+YOLO) → for-llm hattını uçtan-uca yönetir.

2) Sistem/Gereksinimler

Windows 10/11 (PowerShell ile komutlar Windows odaklı; Linux/macOS de çalışır, komutlar uyarlanır)

Python 3.11+ (önerilir)

Tesseract OCR (zorunlu)

YOLO modeli (.pt dosyası)

3) Kurulum (adım adım)
3.1 Python ortamı
# repo klasörüne geç
cd C:\Projects\AgenTest\agent-controller

# sanal ortam (önerilir)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# bağımlılıkları kur
pip install -r requirements.txt

3.2 Tesseract OCR kurulumu

Windows için Tesseract’ı kur (varsayılan yol):
C:\Program Files\Tesseract-OCR\tesseract.exe

Türkçe+İngilizce için dil dosyaları genelde birlikte gelir. Yoksa tessdata altına tur.traineddata + eng.traineddata ekle.

Aşağıdakini .env dosyana koy (reponun kökünde):

# OCR
OCR_ENABLED=True
TESSERACT_LANG=tur+eng
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
OCR_PARALLEL=8
OCR_SYNC_BUDGET=64


Tesseract’ın bin dizini PATH’e ekliyse iyidir; değilse pytesseract.pytesseract.tesseract_cmd kod içinden set edilir. Biz .env ile TESSDATA_PREFIX veriyoruz.

3.3 YOLO modelini indirme

Seçenek A — Hugging Face CLI ile:

# Klasörü hazırla
New-Item -ItemType Directory -Force -Path models\yolo | Out-Null

# HF CLI yoksa kur
pip install "huggingface_hub[cli]"

# Modeli indir (örnek: orasul/deki-yolo)
hf download orasul/deki-yolo --include "best.pt" --local-dir models\yolo

# Bizim konfigde beklenen isim:
Rename-Item models\yolo\best.pt models\yolo\deki-best.pt -Force


Seçenek B — Manuel indirme:
Model dosyasını .pt olarak models\yolo\deki-best.pt konumuna yerleştir.

.env’e YOLO ayarlarını ekle:

# YOLO ikon tespiti
ICON_YOLO_ENABLED=True
ICON_YOLO_MODEL_PATH=models\yolo\deki-best.pt
ICON_YOLO_CONF=0.30
ICON_YOLO_IOU=0.50
ICON_MATCH_MIN_IOU=0.12
ICON_ONLY_ON_EMPTY=True
ICON_DETECT_WHOLE_SCREEN=True
ICON_BOX_MIN_W=6
ICON_BOX_MIN_H=6
ICON_BOX_MAX_W=1600
ICON_BOX_MAX_H=1600
# sınıf eşlemeleri (dedicated class -> etiket)
ICON_CLASS_OVERRIDES=ImageView:icon,Text:text,View:container,Line:separator


Ayar ipuçları:

Daha çok ikon etiketi için ICON_MATCH_MIN_IOU=0.10 deneyebilirsin.

Daha az gürültü için ICON_YOLO_CONF=0.35.

OCR düşük güvenlikli/dup’larda da ikon denensin dersen ICON_ONLY_ON_EMPTY=False.

3.4 SUT URL’si ve genel ayarlar

.env:

# SUT'tan state al
SUT_STATE_URL=http://127.0.0.1:18080/state
SUT_TIMEOUT_SEC=45

# Filtre/Dedupe davranışı
DEDUPE_MODE=name+ct_rank+canon
STRIP_FIELDS=controlType,enabled,idx,patterns

# Genel servis portu
HOST=0.0.0.0
PORT=18800

4) Servisi çalıştırma
# sanal ortam açıksa:
uvicorn agent_controller.main:app --host 0.0.0.0 --port 18800 --reload
# veya .env’de HOST/PORT varsa:
# uvicorn agent_controller.main:app --reload


Başarılı durumda loglarda FastAPI ayaklanır ve endpoints aktif olur.

5) Hızlı testler (PowerShell)
5.1 Konfig/yük durumu
Invoke-RestMethod http://127.0.0.1:18800/debug/icon-yolo
Invoke-RestMethod http://127.0.0.1:18800/config


Beklenen:

icon_yolo_available=True

icon_yolo_model_loaded=True

icon_yolo_model_path=models\yolo\deki-best.pt

5.2 Akış testi (hazır betik)

Repo kökünde bir icon_yolo_test.ps1 varsa:

powershell -ExecutionPolicy Bypass -File .\icon_yolo_test.ps1


Çıktıda şunlar görünür:

OCR istatistikleri: OCR attempted / high / low

YOLO: detections_total, matched

Eşleşen örnekler listesi

5.3 Üç ayrı state çıktısını kaydetme
# RAW (SUT’tan doğrudan)
try {
  $raw = Invoke-RestMethod -Method Post http://127.0.0.1:18080/state -ContentType 'application/json' -Body '{}'
} catch {
  # SUT kapalıysa controller'dan proxy (raw)
  $raw = Invoke-RestMethod -Method Post http://127.0.0.1:18800/state/raw -ContentType 'application/json' -Body '{}'
}

# FILTERED (filtre + OCR + YOLO + _debug)
$filtered = Invoke-RestMethod -Method Post http://127.0.0.1:18800/state/filtered -ContentType 'application/json' -Body '{}'

# FOR-LLM (sade & final paket)
$forllm = Invoke-RestMethod -Method Post http://127.0.0.1:18800/state/for-llm -ContentType 'application/json' -Body '{}'

# Diske yaz
$raw      | ConvertTo-Json -Depth 12 | Out-File -Encoding UTF8 .\state_raw.json
$filtered | ConvertTo-Json -Depth 12 | Out-File -Encoding UTF8 .\state_filtered_debug.json
$forllm   | ConvertTo-Json -Depth 12 | Out-File -Encoding UTF8 .\state_for_llm.json

"Saved: state_raw.json, state_filtered_debug.json, state_for_llm.json"


Hızlı doğrulama:

# eleman sayıları
"Counts -> raw: $($raw.elements.Count) | filtered: $($filtered.elements.Count) | for-llm: $($forllm.elements.Count)"

# for-llm içinde ismi boş + ikon atanmış olanlar
(Get-Content .\state_for_llm.json -Raw | ConvertFrom-Json).elements |
  Where-Object { (-not $_.name) -and $_.name_icon } |
  Select-Object name_icon, name_icon_conf, name_icon_iou, rect


Beklenen:

filtered ≤ raw (dedupe/filtre yüzünden)

for-llm ≈ filtered (sadeleştirilmiş hal; debug alanları yok)

name_icon* alanları, adı boş kalmış bazı öğelerde dolu olmalı.

6) Sağlanan HTTP uçları

POST /state/raw — SUT’tan passtrough ham state (veya SUT kapalıysa proxied ham)

POST /state/filtered — filtre + OCR + YOLO uygulanmış state. _debug altında metrikler:

ocr_attempted, ocr_high, ocr_low, icon_yolo.detections_total, icon_yolo.matched, vb.

POST /state/for-llm — LLM’e gidecek sade paket (seçilmiş alanlar taşınır: name, path, rect, windowActive, varsa name_ocr* ve name_icon*)

GET /debug/icon-yolo — YOLO hazır/yüklü mü, model yolu, class adları

GET /config — Etkin konfig (OCR/YOLO/eşikler/dedupe vb.)

Tüm POST uçları boş gövdeyle çalışır: -Body '{}'.

7) Sık karşılaşılan konular (Troubleshooting)

YOLO “detections_total: 0”

Model yolu doğru mu? (ICON_YOLO_MODEL_PATH)

Eşik çok mu yüksek? ICON_YOLO_CONF’u düşür (örn. 0.30 → 0.25).

Eşleşme toleransı: ICON_MATCH_MIN_IOU’yu 0.12 → 0.10 dene.

Tesseract bulunamadı / dil dosyası hatası

TESSDATA_PREFIX doğru mu? (...Tesseract-OCR\tessdata)

TESSERACT_LANG=tur+eng dil dosyaları mevcut mu?

PATH’te tesseract.exe yoksa, yine de pytesseract bulabilir; ama dosya yolu düzgün olmalı.

PowerShell tek satırda elseif hatası

PS’te elseif bir if { } elseif { } else { } bloğunun parçası olmalı. Yeni satıra tek başına yazılırsa “komut bulunamadı” verir. Çözüm: if bloğunun aynı birleştirilmiş komut içinde kullan.

Performans

OCR thread sayısı: OCR_PARALLEL

Ekran tarama alanı kısıtları: ICON_BOX_MIN/MAX_*

Tam ekran yerine alan kısıtlama gerekirse ICON_DETECT_WHOLE_SCREEN=False yapıp kendi ROI mantığını ekleyebilirsin.

8) Örnek “iyi” bir akış nasıl görünür?

GET /config → icon_yolo_enabled/available/model_loaded = True

POST /state/filtered → _debug.ocr_attempted > 0, _debug.icon_yolo.detections_total > 0, matched ≥ 1

POST /state/for-llm → bazı öğelerde name_icon, name_icon_conf, name_icon_iou dolu; OCR ile ad kazanmış öğeler doğru isimlere sahip.