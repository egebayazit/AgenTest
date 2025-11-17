# OmniParser (agent-ods)

Bu modül, AgenTest için kullanılan OmniParser + Florence2 + YOLO tabanlı ODS servisidir.  
Aşağıdaki adımlar Windows + PowerShell ortamı için özet biçimde sunulmuştur.

---

## 1. Ortam Kurulumu

1. Conda kurulumu yapılır.  
2. Depo klonlanır:

   ```powershell
   git clone https://github.com/egebayazit/AgenTest
   cd AgenTest\agent-ods\OmniParser

3. Ortam oluşturulur ve bağımlılıklar yüklenir:
  
  conda create -n omni python=3.10 -y
  conda activate omni
  pip install -r requirements-ods.txt

## 2. Ağırlık Dosyalarının Hazırlanması

OmniParser/
  weights/
    icon_detect/
      model.pt
    icon_caption_florence/
      config.json
      generation_config.json
      model.safetensors

-model.pt: YOLOv8 tabanlı ikon tespit modeli

-Florence2 modeli: Hugging Face üzerinden indirilen microsoft/Florence-2-base dosyaları

3. Sunucunun Çalıştırılması:

conda activate omni
cd C:\Projects\AgenTest\agent-ods\OmniParser

$env:OMNI_NO_FLASH_STUB = "1"
$env:TRANSFORMERS_NO_FLASH_ATTENTION = "1"
$env:TRANSFORMERS_ATTENTION_IMPLEMENTATION = "sdpa"

python -m omnitool.omniparserserver.omniparserserver `
  --som_model_path .\weights\icon_detect\model.pt `
  --caption_model_name florence2 `
  --caption_model_path .\weights\icon_caption_florence `
  --device cuda

4. Test İsteği

$IMG  = "C:\path\to\your\screenshot.png"
$URL  = "http://127.0.0.1:8000/parse"

$bytes = [System.IO.File]::ReadAllBytes($IMG)
$b64   = [System.Convert]::ToBase64String($bytes)

$body = @{ base64_image = $b64 } | ConvertTo-Json -Compress
$r = Invoke-RestMethod -Uri $URL -Method POST -ContentType "application/json" -Body $body

$r.latency
$r.parsed_content_list | Select-Object -First 5
