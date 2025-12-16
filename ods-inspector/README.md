# Agent Inspector

ODS (Omniparser) tabanlı UI element inspector aracı.

## Gereksinimler
- omniparserserver çalışır durumda olmalı (port 8000)
- Windows 10/11

## Kullanım

1. inspector.exe'yi çalıştır
2. **F4**: Fare imlecinin altındaki elementi bul
3. **F5**: Clipboard'daki isimle eşleşen elementleri bul

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## API Endpoints

- `POST /get-id-from-ods` - Koordinattan element bulma
- `POST /get-coords-from-ods` - Element isminden koordinat bulma
