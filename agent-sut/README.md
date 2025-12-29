# agent-sut
## Derleme ve paketleme rehberi

### On kosullar
- **Windows 10/11** uzerinde Visual Studio Build Tools 2022 (MSVC v143 ve Windows 10 SDK 10.0.26100+).
- **CMake 3.20** veya uzeri.
- **JDK 21** (JAVA_HOME ayarlanmis olmali) ve `jlink` araci.
- **Apache Maven 3.9+**.
- Powershell veya Git Bash ile calisan build ortami.

### Tum SUT'u CMake ile derlemek 
```powershell
cd AgenTest\agent-sut/web

pyinstaller --onefile --name web_agent web_agent.py
cd ..
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
Bu adimlar Maven jarini (`jvm_agent_jar` hedefi) ve trimmed JVM runtime'ini (`jvm_runtime_zip`) otomatik olarak olusturur. Sonuc exe `build\Release\agent_sut.exe` konumuna yazilir.

### Java kaynak dosyalari degistikten sonra JAR'i yeniden derlemek
Eger `jvm/src` altindaki Java dosyalarini degistirdiyseniz, CMake incremental build JAR'i guncellemeyebilir. Bu durumda:
```powershell
cmake --build build --config Release --target jvm_agent_jar --clean-first
cmake --build build --config Release
```

# web çalıstırma
```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome_debug_profile"
```