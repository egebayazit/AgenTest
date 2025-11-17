# OmniParser (ODS)

Bu klasör, ekran görüntülerinden ikon + metin yakalayıp kutulu çıktı üreten OmniParser servisidir.  
FastAPI + Uvicorn ile bir HTTP servisi olarak çalışır.

## 1. Ön Koşullar

- Git
- [Miniconda / Anaconda](https://www.anaconda.com/) (önerilen)
- NVIDIA GPU (opsiyonel ama tavsiye edilir)
- Windows PowerShell (komutlar PowerShell formatında)

> Not: CPU ile de çalışabilir ama Florence2 sebebiyle **çok yavaş** olabilir.

---

## 2. Depoyu Klonla

```powershell
git clone <REPO_URL>
cd <repo-kökü>\agent-ods\OmniParser
OmniParser klasörü içinde şunlar olmalı:

omnitool/omniparserserver/omniparserserver.py

util/omniparser.py, util/utils.py

weights/icon_detect/

weights/icon_caption_florence/

requirements-ods.txt

3. Conda Ortamı Oluştur
powershell
Kodu kopyala
conda create -n ods2 python=3.10 -y
conda activate ods2
4. PyTorch + Bağımlılıkları Kur
4.1. PyTorch (CUDA’lı veya CPUsu)
Tam komut için kendi sistemine uygun olanı pytorch.org üzerinden seçmen en doğrusu.

Örnek (CUDA destekliyse):

powershell
Kodu kopyala
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
Veya sadece CPU:

powershell
Kodu kopyala
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
4.2. Diğer paketler
powershell
Kodu kopyala
cd <repo-kökü>\agent-ods\OmniParser
pip install -r requirements-ods.txt
5. Ağırlıklar (weights) Klasörü
Aşağıdaki dosyaların mevcut olduğundan emin ol:

YOLO ikon dedektörü:

weights/icon_detect/model.pt

weights/icon_detect/model.yaml

Florence2 caption modeli:

weights/icon_caption_florence/model.safetensors

weights/icon_caption_florence/config.json

weights/icon_caption_florence/generation_config.json

Eğer repo ile beraber gelmediyse:

Kendi YOLO modelini weights/icon_detect/model.pt olarak koyabilirsin.

Kendi Florence2 checkpoint’ini weights/icon_caption_florence/ klasörüne aynı isimlerle yerleştirmelisin.

6. Sunucuyu Başlatma
OmniParser klasöründe ve ods2 ortamı aktifken:

powershell
Kodu kopyala
conda activate ods2
cd <repo-kökü>\agent-ods\OmniParser

# Flash-Attn kapalı, SDPA açık
$env:OMNI_NO_FLASH_STUB = "1"
$env:TRANSFORMERS_NO_FLASH_ATTENTION = "1"
$env:TRANSFORMERS_ATTENTION_IMPLEMENTATION = "sdpa"

python -m omnitool.omniparserserver.omniparserserver `
  --som_model_path .\weights\icon_detect\model.pt `
  --caption_model_name florence2 `
  --caption_model_path .\weights\icon_caption_florence `
  --device cuda
GPU’n yoksa --device cpu kullanabilirsin (daha yavaş çalışır):

powershell
Kodu kopyala
python -m omnitool.omniparserserver.omniparserserver `
  --som_model_path .\weights\icon_detect\model.pt `
  --caption_model_name florence2 `
  --caption_model_path .\weights\icon_caption_florence `
  --device cpu
Sunucu açıldığında Uvicorn loglarında şunu görmelisin:

text
Kodu kopyala
Omniparser initialized!!!
Uvicorn running on http://127.0.0.1:8000
7. Hızlı Test
Ayrı bir PowerShell penceresinde:

powershell
Kodu kopyala
# Sunucu ayakta mı?
Invoke-WebRequest -Uri "http://127.0.0.1:8000/probe"

# Örnek parse isteği
$IMG  = "C:\path\to\your\screenshot.png"
$URL  = "http://127.0.0.1:8000/parse"

$bytes = [System.IO.File]::ReadAllBytes($IMG)
$b64   = [System.Convert]::ToBase64String($bytes)

$body = @{ base64_image = $b64 } | ConvertTo-Json -Compress
$r = Invoke-RestMethod -Uri $URL -Method POST -ContentType "application/json" -Body $body

$r.latency
$r.parsed_content_list | Select-Object -First 5
Her şey doğruysa:

HTTP 200 dönmeli

Log’da start parsing... ve time: X satırlarını görmelisin

parsed_content_list içinde ikon/text kutuları görünür.