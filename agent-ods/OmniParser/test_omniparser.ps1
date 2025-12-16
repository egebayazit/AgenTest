# Test Script for OmniParser
$URL = "http://127.0.0.1:8000/parse"
$OUTPUT_DIR = "C:\Users\hazal\Desktop\deneme"

# Dosyayi otomatik bul
$IMG = Get-ChildItem "C:\Users\hazal\Desktop\deneme\*.png" | Where-Object { $_.Name -like "*Ekran*100106*" } | Select-Object -First 1 -ExpandProperty FullName

if (-not $IMG -or -not (Test-Path $IMG)) {
    Write-Host "HATA: Goruntu dosyasi bulunamadi!" -ForegroundColor Red
    Write-Host "Lutfen dosya yolunu kontrol edin veya dosyayi yeniden adlandirin." -ForegroundColor Yellow
    exit 1
}

Write-Host "=== OmniParser Test Baslatiliyor ===" -ForegroundColor Green
Write-Host "Goruntu: $IMG" -ForegroundColor Cyan
Write-Host "API URL: $URL" -ForegroundColor Cyan
Write-Host ""

# 1. Goruntuyu Base64'e cevir
Write-Host "1. Goruntu okunuyor..." -ForegroundColor Yellow
try {
    $bytes = [System.IO.File]::ReadAllBytes($IMG)
    $b64 = [System.Convert]::ToBase64String($bytes)
    Write-Host "   OK Goruntu basariyla okundu (Boyut: $($bytes.Length) bytes)" -ForegroundColor Green
} catch {
    Write-Host "   X HATA: Goruntu okunamadi - $_" -ForegroundColor Red
    exit 1
}

# 2. API'ye istek gonder
Write-Host "2. API'ye istek gonderiliyor..." -ForegroundColor Yellow
$body = @{ base64_image = $b64 } | ConvertTo-Json -Compress

try {
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri $URL -Method POST -ContentType "application/json" -Body $body
    $stopwatch.Stop()
    Write-Host "   OK API yaniti alindi (Sure: $($stopwatch.Elapsed.TotalSeconds) saniye)" -ForegroundColor Green
} catch {
    Write-Host "   X HATA: API yanit vermedi - $_" -ForegroundColor Red
    exit 1
}

# 3. Latency bilgisi
Write-Host ""
Write-Host "=== Performans Bilgisi ===" -ForegroundColor Green
Write-Host "API Latency: $($response.latency) saniye" -ForegroundColor Cyan

# 4. Parsed content ozeti
Write-Host ""
Write-Host "=== Parse Edilen Icerik Ozeti ===" -ForegroundColor Green
$textCount = ($response.parsed_content_list | Where-Object { $_.type -eq "text" }).Count
$iconCount = ($response.parsed_content_list | Where-Object { $_.type -eq "icon" }).Count
Write-Host "Toplam Text Box: $textCount" -ForegroundColor Cyan
Write-Host "Toplam Icon Box: $iconCount" -ForegroundColor Cyan
Write-Host "Toplam Eleman: $($response.parsed_content_list.Count)" -ForegroundColor Cyan

# 5. Ilk 10 elemani goster
Write-Host ""
Write-Host "=== Ilk 10 Parse Edilen Eleman ===" -ForegroundColor Green
$response.parsed_content_list | Select-Object -First 10 | ForEach-Object -Begin { $i = 0 } -Process {
    $i++
    Write-Host "[$i] Type: $($_.type) | Content: $($_.content)" -ForegroundColor White
    Write-Host "    BBox: [$($_.bbox[0]), $($_.bbox[1]), $($_.bbox[2]), $($_.bbox[3])]" -ForegroundColor DarkGray
    Write-Host "    Source: $($_.source)" -ForegroundColor DarkGray
    Write-Host ""
}

# 6. Full JSON response'u kaydet
Write-Host "=== Sonuclar Kaydediliyor ===" -ForegroundColor Green
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Full JSON kaydet
$jsonPath = Join-Path $OUTPUT_DIR "omniparser_response_$timestamp.json"
try {
    $response | ConvertTo-Json -Depth 10 | Out-File -FilePath $jsonPath -Encoding UTF8
    Write-Host "   OK Full JSON kaydedildi: $jsonPath" -ForegroundColor Green
} catch {
    Write-Host "   X HATA: JSON kaydedilemedi - $_" -ForegroundColor Red
}

# 7. Overlay image'i kaydet
if ($response.image) {
    $overlayPath = Join-Path $OUTPUT_DIR "omniparser_overlay_$timestamp.png"
    try {
        $imageBytes = [System.Convert]::FromBase64String($response.image)
        [System.IO.File]::WriteAllBytes($overlayPath, $imageBytes)
        Write-Host "   OK Overlay goruntusu kaydedildi: $overlayPath" -ForegroundColor Green
    } catch {
        Write-Host "   X HATA: Overlay goruntusu kaydedilemedi - $_" -ForegroundColor Red
    }
} else {
    Write-Host "   ! Uyari: Response'da overlay image bulunamadi" -ForegroundColor Yellow
}

# 8. Ozet rapor olustur
$reportPath = Join-Path $OUTPUT_DIR "omniparser_report_$timestamp.txt"
try {
    $textBoxes = $response.parsed_content_list | Where-Object { $_.type -eq "text" } | ForEach-Object { "- $($_.content)" }
    $iconBoxes = $response.parsed_content_list | Where-Object { $_.type -eq "icon" } | ForEach-Object { "- $($_.content)" }
    
    $report = @"
=== OmniParser Test Raporu ===
Test Zamani: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Goruntu: $IMG
API URL: $URL

=== Performans ===
API Latency: $($response.latency) saniye
Toplam Islem Suresi: $($stopwatch.Elapsed.TotalSeconds) saniye

=== Icerik Ozeti ===
Toplam Text Box: $textCount
Toplam Icon Box: $iconCount
Toplam Eleman: $($response.parsed_content_list.Count)

=== Kaydedilen Dosyalar ===
- JSON Response: $jsonPath
- Overlay Image: $overlayPath

=== Text Box Icerikleri ===
$($textBoxes -join "`n")

=== Icon Box Icerikleri ===
$($iconBoxes -join "`n")
"@
    
    $report | Out-File -FilePath $reportPath -Encoding UTF8
    Write-Host "   OK Test raporu kaydedildi: $reportPath" -ForegroundColor Green
} catch {
    Write-Host "   X HATA: Rapor kaydedilemedi - $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Test Tamamlandi ===" -ForegroundColor Green
Write-Host ""

# 9. Dosyalari ac (opsiyonel)
$openFiles = Read-Host "Kaydedilen dosyalari acmak ister misiniz? (E/H)"
if ($openFiles -eq "E" -or $openFiles -eq "e") {
    if (Test-Path $jsonPath) { Start-Process notepad $jsonPath }
    if (Test-Path $overlayPath) { Start-Process $overlayPath }
    if (Test-Path $reportPath) { Start-Process notepad $reportPath }
}