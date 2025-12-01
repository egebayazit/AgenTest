# OmniParser (agent-ods) - v2.0

Bu modül, AgenTest projesi için kullanılan **OmniParser v2.0** (YOLO v2.0 + Florence-2) tabanlı Optik Algılama Servisidir (ODS).

## 1. Ortam Kurulumu (Windows + PowerShell)

1. **Conda Ortamı Oluşturma:**
   ```powershell
   conda create -n omni python=3.10 -y
   conda activate omni
Bağımlılıkları Yükleme: Proje dizinindeyken gerekli kütüphaneleri yükleyin:

PowerShell

cd agent-ods\OmniParser
pip install -r requirements.txt
# v2.0 uyumluluğu için güncellemeler:
pip install -U ultralytics transformers torch torchvision
2. Ağırlık Dosyalarının Hazırlanması (Weights Setup)
OmniParser v2.0, büyük model dosyalarına ihtiyaç duyar. Bu dosyalar GitHub'a yüklenmez, manuel indirilmelidir.

Klasör yapısı şöyle olmalıdır:

Plaintext

OmniParser/
└── weights/
    ├── icon_detect/
    │   ├── model.pt      (YOLO v2.0)
    │   └── model.yaml
    └── icon_caption_florence/
        ├── config.json
        ├── model.safetensors (Florence-2)
        ├── tokenizer.json
        └── ... (diğer tokenizer dosyaları)
İndirme Komutları (PowerShell)
Aşağıdaki komutları sırasıyla çalıştırarak gerekli klasörleri oluşturun ve dosyaları indirin.

1. Klasörleri Oluştur:

PowerShell

New-Item -ItemType Directory -Force -Path weights/icon_detect
New-Item -ItemType Directory -Force -Path weights/icon_caption_florence
2. Icon Detect (YOLO) Modellerini İndir:

PowerShell

cd weights/icon_detect
Invoke-WebRequest -Uri "[https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt)" -OutFile "model.pt"
Invoke-WebRequest -Uri "[https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.yaml](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.yaml)" -OutFile "model.yaml"
cd ../..
3. Icon Caption (Florence-2) Modellerini İndir:

PowerShell

cd weights/icon_caption_florence
# Temel Model Dosyaları (OmniParser v2.0 Reposundan)
Invoke-WebRequest -Uri "[https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/config.json](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/config.json)" -OutFile "config.json"
Invoke-WebRequest -Uri "[https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/generation_config.json](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/generation_config.json)" -OutFile "generation_config.json"
Invoke-WebRequest -Uri "[https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/model.safetensors](https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/model.safetensors)" -OutFile "model.safetensors"

# Tokenizer Dosyaları (Eksikler Florence-2-base Reposundan tamamlanır)
Invoke-WebRequest -Uri "[https://huggingface.co/microsoft/Florence-2-base/resolve/main/tokenizer_config.json](https://huggingface.co/microsoft/Florence-2-base/resolve/main/tokenizer_config.json)" -OutFile "tokenizer_config.json"
Invoke-WebRequest -Uri "[https://huggingface.co/microsoft/Florence-2-base/resolve/main/vocab.json](https://huggingface.co/microsoft/Florence-2-base/resolve/main/vocab.json)" -OutFile "vocab.json"
Invoke-WebRequest -Uri "[https://huggingface.co/microsoft/Florence-2-base/resolve/main/tokenizer.json](https://huggingface.co/microsoft/Florence-2-base/resolve/main/tokenizer.json)" -OutFile "tokenizer.json"
cd ../..
3. Sunucuyu Başlatma
Flash Attention sorunlarını önlemek için ortam değişkenlerini ayarlayıp sunucuyu başlatın:

PowerShell

$env:OMNI_NO_FLASH_STUB = "1"
$env:TRANSFORMERS_NO_FLASH_ATTENTION = "1"
$env:TRANSFORMERS_ATTENTION_IMPLEMENTATION = "sdpa"

python -m omnitool.omniparserserver.omniparserserver `
  --som_model_path .\weights\icon_detect\model.pt `
  --caption_model_name florence2 `
  --caption_model_path .\weights\icon_caption_florence `
  --device cuda
Sunucu http://127.0.0.1:8000 adresinde çalışacaktır.

4. Test İsteği Örneği
Başka bir PowerShell penceresinden test etmek için:

PowerShell

$IMG  = "imgs/test.jpg" # Test edilecek resim yolu
$URL  = "[http://127.0.0.1:8000/parse/](http://127.0.0.1:8000/parse/)"

# Resmi Base64'e çevir
$bytes = [System.IO.File]::ReadAllBytes($IMG)
$b64   = [System.Convert]::ToBase64String($bytes)

# İsteği Gönder
$body = @{ base64_image = $b64 } | ConvertTo-Json -Compress
$r = Invoke-RestMethod -Uri $URL -Method POST -ContentType "application/json" -Body $body

# Sonucu Görüntüle
Write-Host "Latency: $($r.latency)"
$r.parsed_content_list | Select-Object -First 5