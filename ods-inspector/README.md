# ODS + WinDriver Inspector

UI element inspector aracı - ODS (Omniparser) ve Windows UI Automation desteği.

## Özellikler

- **F3**: Element bul (hover yok - fare köşeye taşınır, screenshot alınır)
- **F4**: Element bul (hover dahil - mevcut durum)
- **F5**: Clipboard'daki isimle element bul (ODS)

## Çıktı Formatı

```
[ODS]
  name : "Button Text"
  id   : 42

[WINDRIVER]
  name  : "Button Text"
  id    : "btnSubmit"
  value : ""
```

## Kullanım

### SUT PC'de (Varsayılan)
```powershell
.\inspector.exe
# ODS Server: http://10.182.6.60:8000
```

### Lokal Test (Kendi Makinende)
```powershell
$env:ODS_HOST = "localhost"
.\inspector.exe
# ODS Server: http://localhost:8000
```

### Farklı Ağdan Bağlanma (Örn: Mobil Hotspot)
ODS sunucusu farklı bir IP'de çalışıyorsa:
```powershell
$env:ODS_HOST = "10.228.19.178"
.\inspector.exe
# ODS Server: http://10.228.19.178:8000
```

Veya tek satırda:
```powershell
$env:ODS_HOST="10.228.19.178"; .\inspector.exe
```

> **Not:** Ağ değişikliğinde "ODS server not reachable" hatası alırsan, ODS sunucusunun yeni IP'sini (`ipconfig` ile) bul ve `ODS_HOST` değişkenini güncelle.

## Gereksinimler
- OmniParser server çalışır durumda (port 8000)
- Windows 10/11

## Build

```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

Çıktı: `build/Release/inspector.exe`
