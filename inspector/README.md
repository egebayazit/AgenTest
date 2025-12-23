# Inspector (ODS + Win + JVM)

UI element inspector - ODS (Omniparser), Windows UI Automation ve JVM (Swing/JavaFX) desteği.

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

[JVM]
  id    : "W0.100.0.2"
  class : "JButton"
  text  : "Submit"
```

## Kullanım

### Varsayılan (Lokal ODS)
```powershell
.\inspector.exe
# ODS Server: http://localhost:8000
```

### Uzak ODS Sunucusuna Bağlanma
```powershell
$env:ODS_HOST = "..."
.\inspector.exe
# ODS Server: http://.......:8000
```

### JVM Desteği
JVM desteği için:
1. Sistemde Java kurulu olmalı (`JAVA_HOME` veya PATH'de)
2. `jvm` klasörü inspector.exe yanında olmalı:
```
inspector/
├── inspector.exe
└── jvm/
    └── target/jvm-element-finder-*.jar  # Java agent
```

Veya `JVM_DIR` environment değişkeni ile yol belirtilebilir:
```powershell
$env:JVM_DIR = "C:\path\to\jvm"
.\inspector.exe
```

## Gereksinimler
- OmniParser server çalışır durumda (port 8000)
- Windows 10/11
- JVM desteği için: Java runtime ve jvm-element-finder jar

## Build

```powershell
cd inspector
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
### Java kaynak dosyalari degistikten sonra JAR'i yeniden derlemek
Eger `jvm/src` altindaki Java dosyalarini degistirdiyseniz, CMake incremental build JAR'i guncellemeyebilir. Bu durumda:
```powershell
cmake --build build --config Release --target jvm_agent_jar --clean-first
cmake --build build --config Release
```
Çıktı: `build/Release/inspector.exe`
