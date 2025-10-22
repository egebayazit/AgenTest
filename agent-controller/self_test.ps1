Param(
  [string]$ControllerUrl = "http://127.0.0.1:18800",
  [string]$SutUrl        = "http://127.0.0.1:18080",
  [string]$ModelPath     = "models\yolo\deki-best.pt",
  [string]$TessdataPath  = "C:\Program Files\Tesseract-OCR\tessdata"
)

$ErrorActionPreference = "Stop"

function fail($msg) {
  Write-Host "`n[FAIL] $msg" -ForegroundColor Red
  exit 1
}

function ok($msg) {
  Write-Host "[OK] $msg" -ForegroundColor Green
}

function warn($msg) {
  Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

Write-Host "=== Agent-Controller Self Test ===`n"

# ---------------- helpers ----------------
function Get-StateType($raw) {
  try {
    if ($null -ne $raw.state_type) {
      $st = "$($raw.state_type)".ToLower()
      if ($st -eq "jvm") { return "JVM" }
      if ($st -eq "windows") { return "Windows" }
    }
    if ($null -ne $raw.componentCount) { return "JVM" }
    if ($null -ne $raw.b64 -and ($raw.b64 -is [string]) -and $raw.b64.Length -gt 1000) { return "JVM" }

    $hasScreen = $null -ne $raw.screen -and $null -ne $raw.screen.w -and $null -ne $raw.screen.h
    $hasElements = $null -ne $raw.elements
    if ($hasElements) {
      $first = $raw.elements | Select-Object -First 1
      if ($null -ne $first) {
        # JVM ağaçlarında children/class çok yaygındır
        if ($null -ne $first.children -or $null -ne $first.class) { return "JVM" }
        # Windows’ta tipik olarak rect bulunur
        if ($null -ne $first.rect -and $null -ne $first.rect.l) { return "Windows" }
      }
    }
    if ($hasScreen) { return "Windows" }
  } catch { }

  return "Unknown"
}

function Has-OnlyKeys($obj, $allowedKeys) {
  $keys = @()
  if ($obj -is [System.Collections.IDictionary]) {
    $keys = $obj.Keys
  } else {
    $keys = ($obj | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name)
  }
  foreach ($k in $keys) {
    if (-not ($allowedKeys -contains $k)) { return $false }
  }
  return $true
}

function Validate-ForLlm-Schema($forllm) {
  $allowed = @("name","center","path")
  $elems = $forllm.elements
  if ($null -eq $elems) { fail "for-llm: elements alanı yok." }
  $viol = 0
  foreach ($e in $elems) {
    if (-not (Has-OnlyKeys $e $allowed)) {
      $viol++
    }
    if ($null -eq $e.name -or [string]::IsNullOrWhiteSpace([string]$e.name)) {
      warn "for-llm: boş isimli bir eleman bulundu (LLM'e gönderilmeyecekti)."
      return $false
    }
    if ($null -eq $e.center -or $null -eq $e.center.x -or $null -eq $e.center.y) {
      warn "for-llm: center alanı eksik görünüyor."
      return $false
    }
    if ($null -eq $e.path) {
      warn "for-llm: path alanı eksik görünüyor."
      return $false
    }
  }
  if ($viol -gt 0) {
    fail "for-llm: elementlerde izin verilmeyen ekstra alan(lar) var (yalnızca name, center, path olmalı). İhlal sayısı: $viol"
  }
  return $true
}

# 0) Dosya/ortam ön kontrolü
if (-not (Test-Path ".env")) { fail ".env bulunamadı." } else { ok ".env var" }
if (-not (Test-Path $ModelPath)) { fail "YOLO model yok: $ModelPath" } else { ok "YOLO model var: $ModelPath" }
if (-not (Test-Path $TessdataPath)) { fail "Tesseract tessdata yok: $TessdataPath" } else { ok "Tesseract tessdata var: $TessdataPath" }

# 1) Servis ayakta mı?
try {
  $cfg = Invoke-RestMethod "$ControllerUrl/config"
  ok "/config yanıt verdi"
} catch { fail "/config erişilemedi: $($_.Exception.Message)" }

# 2) YOLO debug ve yük durumu
try {
  $ydbg = Invoke-RestMethod "$ControllerUrl/debug/icon-yolo"
  if (-not $ydbg.available -or -not $ydbg.enabled) { fail "YOLO available/enabled False görünüyor." }
  if (-not $ydbg.loaded) { fail "YOLO model loaded=False. Model yolu: $($ydbg.model_path)" }
  if ($ydbg.model_path -ne $ModelPath) { warn "model_path farklı: $($ydbg.model_path)" }
  ok "YOLO yüklü ve aktif"
} catch { fail "/debug/icon-yolo çağrısı başarısız: $($_.Exception.Message)" }

# 3) RAW/FILTERED/FOR-LLM çek
try {
  try {
    $raw = Invoke-RestMethod -Method Post "$SutUrl/state" -ContentType 'application/json' -Body '{}'
    ok "RAW SUT üzerinden alındı"
  } catch {
    $raw = Invoke-RestMethod -Method Post "$ControllerUrl/state/raw" -ContentType 'application/json' -Body '{}'
    ok "RAW controller proxy üzerinden alındı"
  }

  $filtered = Invoke-RestMethod -Method Post "$ControllerUrl/state/filtered" -ContentType 'application/json' -Body '{}'
  $forllm   = Invoke-RestMethod -Method Post "$ControllerUrl/state/for-llm" -ContentType 'application/json' -Body '{}'
  ok "filtered & for-llm alındı"
} catch { fail "state çağrılarından biri başarısız: $($_.Exception.Message)" }

# 3.a) RAW tipini belirleyip yazdır
$rawType = Get-StateType $raw
Write-Host ("RAW state_type (heuristic): {0}" -f $rawType)

# 4) Sayı ve metrik kontrolleri (temel)
$rawCount      = ($raw.elements      | Measure-Object).Count
$filteredCount = ($filtered.elements | Measure-Object).Count
$forllmCount   = ($forllm.elements   | Measure-Object).Count

Write-Host ("Counts -> raw: {0} | filtered: {1} | for-llm: {2}" -f $rawCount,$filteredCount,$forllmCount)

# Not: JVM raw.elements genelde root düğümleridir; normalize sonrası filtered sayısı raw'dan çok olabilir.
if ($rawType -eq "Windows") {
  if ($filteredCount -gt $rawCount) { fail "Windows modunda: filtered eleman sayısı raw'dan fazla (beklenmez)." }
} else {
  if ($filteredCount -lt 1) { fail "JVM modunda: filtered boş görünüyor." }
}

if ($forllmCount -lt 1) { fail "for-llm boş görünüyor." }

# 5) OCR + YOLO debug metrikleri
if ($filtered._debug) {
  $ocrAtt = 0; $ocrHi = 0; $yDet = 0; $yMatch = 0
  try { $ocrAtt = [int]$filtered._debug.ocr_attempted } catch {}
  try { $ocrHi  = [int]$filtered._debug.ocr_high } catch {}
  try { $yDet   = [int]$filtered._debug.icon_yolo.detections_total } catch {}
  try { $yMatch = [int]$filtered._debug.icon_yolo.matched } catch {}

  Write-Host ("OCR attempted/high: {0}/{1}" -f $ocrAtt,$ocrHi)
  Write-Host ("YOLO detections/matched: {0}/{1}" -f $yDet,$yMatch)

  if ($ocrAtt -lt 1) { fail "OCR hiç denenmemiş gibi görünüyor." }
  if ($yDet -lt 1)   { fail "YOLO detections_total = 0 (eşik ayarlarına bakın)." }
} else {
  warn "filtered._debug yok; debug metrikleri görünmüyor."
}

# 6) for-LLM şema kontrolü (YALNIZ name, center, path)
if (Validate-ForLlm-Schema $forllm) {
  ok "for-llm şema uygun: sadece name, center, path var"
}

# 7) Örnek ilk 5 eleman
Write-Host "`nfor-llm örnek (ilk 5):"
$forllm.elements | Select-Object -First 5 name, center, path | Format-Table -AutoSize

ok "Self-test tamam: temel akış (raw -> normalized(if JVM) -> filtered(OCR+YOLO) -> for-llm) çalışıyor"
exit 0
