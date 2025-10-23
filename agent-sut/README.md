# agent-sut

`agent-sut` Windows uzerinde calisan System Under Test (SUT) sunucusudur. Tek bir exe icinde uc ana yetenegi birlestirir:
- HTTP uzerinden eylem uygulama (fare/klavye) icin `action` modulu,
- Windows UI Automation ile ekran durumu toplama icin `win` yardimcilari,
- Java UI uygulamalari icin JVM heap state ve ekran yakalama destegi.

Sunucu 0.0.0.0:18080 adresinde dinler ve controller tarafindan /action, /state ve /jvmstate istekleri ile kullanilir.

## Dizin yerlesimi
```
agent-sut/
  CMakeLists.txt          # Tum projeyi (native + JVM) derleyen ana CMake betigi
  server_main.cpp         # HTTP sunucu giris noktasi
  action/                 # Input enjeksiyon kutuphanesi ve opsiyonel mini server
  win/                    # Windows state alma yardimcilari (UIA + ekran yakalama + base64)
  jvm/                    # JVM koprusu, Maven projesi ve jlink ile uretilen runtime
  third_party/            # Tek header bagimliliklar (httplib, nlohmann-json)
  headers.txt             # Ornek HTTP yaniti (Fiddler/Postman icin)
  build/                  # CMake ile olusan build agaci (git tarafindan yok sayilir)
```

### Onemli dosyalar ve klasorler

- `server_main.cpp`: Tum HTTP endpointlerini kaydeden ana exe. `action::ActionHandler` ve `UiaSession` nesneleri ile calisir, ayri log klasorlerine JSON + PNG yazar.
- `action/action_handler.cpp|.h`: `/action` istegindeki JSON planini okur, dogrular ve `ActionRobot` araciligi ile Win32 `SendInput` cagrilarini tetikler.
- `action/action_robot.cpp|.h`: Fare ve klavye seviyesindeki islemler; DPI farklarini giderir, modifier kombinasyonlarini yonetir.
- `action/main_action.cpp`: `BUILD_SUT_ACTION_SERVER=ON` flagi ile tek basina `/action` ve `/healthz` sunucusu uretmek icin kullanilabilir.
- `action/CMakeLists.txt`: Modulu ayri derlemek icin kullanilan minimal betik, artik tekil `../third_party/` dizinini include eder.
- `win/uia_utils.*`: UI Automation (Microsoft UIA) baglantisi, eleman agaclarini toplar, filtreler ve JSON icin ilkel tur donusturur.
- `win/capture.*`: Ekranin PNG olarak yakalanmasi ve ham verinin `std::vector<uint8_t>` olarak donmesi.
- `win/base64.*`: PNG verisini encode/decode etmek icin basit base64 yardimcilari.
- `jvm/jvm_bridge.*`: Native taraftan JVM agent jarini cagiran kopru. `CaptureSnapshot` ile hedef PID uzerinden JVM durumu toplar.
- `jvm/pom.xml`: Maven projesi; `jvm-element-finder-1.0-SNAPSHOT-jar-with-dependencies.jar` cikti jarini uretir.
- `jvm/jvm_agent.rc.in`: Jar ve runtime yolunu native kaynaklara dahil etmek icin CMake tarafindan doldurulan template.
- `jvm/runtime/`, `jvm/runtime.zip`: `jlink` ile olusan trimmed JRE. CMake hedefi `jvm_runtime_zip` tarafindan uretilir.
- `third_party/httplib.h`, `third_party/json.hpp`: cpp-httplib ve nlohmann/json tek header kutuphaneleri.

## HTTP API

### GET /healthz
- Amac: Servisin ayakta oldugunu dogrulamak icin basit saglik kontrolu.
- Yanit: `{"status":"ok"}` JSON.
- Ornek: `curl http://127.0.0.1:18080/healthz`

### POST /action
- Amac: LLM/Controller tarafindan gelen eylem planini uygulamak.
- Beklenen icerik: `Content-Type: application/json`. Govde `action/action_handler.cpp` icindeki formata (action_id, steps, vb.) uymali.
- Donus: `ActionHandler::Handle` sonucunda uretilen JSON (status, applied, hata kodlari).
- Not: Yanitta state bulunmaz; islem sonrasi `/state` ya da `/winstate` cagirilmalidir.
- Ornek:
  ```powershell
  curl -X POST http://127.0.0.1:18080/action `
       -H "Content-Type: application/json" `
       -d "{\"steps\":[{\"type\":\"click\",\"button\":\"left\",\"target\":{\"point\":{\"x\":200,\"y\":200}}}]}"
  ```

### POST /winstate
- Amac: Yalnizca Windows UI Automation durumunu ve ekran goruntusunu almak.
- Donus: JSON icinde `stateType: "windows"`, `screen`, `elements` listesi ve `screenshot.format`. PNG kendisi disk uzerine `winstate_screenshots/` altina kaydedilir.
- Loglar: `winstate_logs/` klasorune timestamp'li JSON, `winstate_screenshots/` klasorune PNG.

### POST /state
- Amac: JVM durumunu tercih ederek kombine state saglamak.
- Ilerleyis:
  1. Varsayilan olarak `jvm_bridge::CaptureSnapshot` ile calisan JVM uygulamalarindan state almaya calisir.
  2. JVM tarafinda hata olursa otomatik olarak `winstate` dondurulur ve yanit JSON icine `fallbackReason` eklenir.
- Donus: `stateType: "jvm"` ya da `"windows"`.
- Loglar: `state_logs/` (JSON) ve `state_screenshots/` (PNG, eger yakalandiysa).

### POST /jvmstate
- Amac: JVM uygulamalarinin durumunu dogrudan yakalamak veya belirli PID icin hedeflemek.
- Parametreler:
  - Query veya JSON govdede `pid` (opsiyonel) belirtebilirsiniz.
- Basarili Yanit: `stateType: "jvm"` ve agentin urettigi alanlar (heap dump, b64 screenshot vb.).
- Hata Yaniti (HTTP 400/500): `code` alanlari `INVALID_PID`, `INVALID_JSON`, `JVM_CAPTURE_FAILED`, `JVM_RESPONSE_PARSE_FAILED`.
- Loglar: `jvmstate_logs/` (tum JSON yanitlar), `jvmstate_screenshots/` (PNG, eger `b64` saglanirsa).

## Derleme ve paketleme rehberi

### On kosullar
- **Windows 10/11** uzerinde Visual Studio Build Tools 2022 (MSVC v143 ve Windows 10 SDK 10.0.26100+).
- **CMake 3.20** veya uzeri.
- **JDK 21** (JAVA_HOME ayarlanmis olmali) ve `jlink` araci.
- **Apache Maven 3.9+**.
- Powershell veya Git Bash ile calisan build ortami.

### Tum projeyi CMake ile derlemek (onerilen)
```powershell
cd C:\Users\hazal\Desktop\AgenTest\agent-sut
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
Bu adimlar Maven jarini (`jvm_agent_jar` hedefi) ve trimmed JVM runtime'ini (`jvm_runtime_zip`) otomatik olarak olusturur. Sonuc exe `build\Release\agent_sut.exe` konumuna yazilir.

### Servisi calistirmak
```powershell
cd C:\Users\hazal\Desktop\AgenTest\agent-sut
.\build\Release\agent_sut.exe
```
Exe calisirken ayni klasorde `winstate_logs/`, `state_logs/`, `jvmstate_logs/` gibi dizinler otomatik olusur.

### JVM agent jarini manuel olarak derlemek
```powershell
cd C:\Users\hazal\Desktop\AgenTest\agent-sut\jvm
mvn -DskipTests package
```
Bu komut `target\jvm-element-finder-1.0-SNAPSHOT-jar-with-dependencies.jar` dosyasini uretir. CMake disinda degisiklik yapilacaksa `pom.xml` icindeki artifact ismini guncellediginizden emin olun.

### jlink ile trimmed JVM runtime olusturmak
CMake otomatik tetikler; manuel olarak calistirmak isterseniz:
```powershell
cd C:\Users\hazal\Desktop\AgenTest\agent-sut\jvm
set JAVA_HOME=C:\Path\To\JDK21
jlink --add-modules java.base,java.instrument,java.management,java.desktop,java.logging,java.datatransfer,java.naming,jdk.attach,jdk.management,jdk.unsupported --strip-java-debug-attributes --no-header-files --no-man-pages --output runtime
Compress-Archive -Path runtime\* -DestinationPath runtime.zip -Force
```
Native build sirasinda CMake bu zip yolunu `jvm_agent.rc` icine yazar.

### Action modulunu ayri test etmek (opsiyonel)
```powershell
cd C:\Users\hazal\Desktop\AgenTest\agent-sut\action
cmake -S . -B build -DBUILD_SUT_ACTION_SERVER=ON
cmake --build build --config Release
.\build\Release\sut_action_server.exe
```
Bu exe yalnizca `/action` ve `/healthz` endpointlerini saglar; kontrolcu entegrasyonunu test etmek icin kullanisli olabilir.

## Loglar ve ciktilar
- `winstate_logs/` ve `winstate_screenshots/`: `/winstate` isteklerinden kaydedilen JSON + PNG.
- `state_logs/` ve `state_screenshots/`: `/state` istegi sonucunda olusan durumlar.
- `jvmstate_logs/` ve `jvmstate_screenshots/`: `/jvmstate` yanitlari ve varsa ekran goruntuleri.
- `jvm/target/`: Maven tarafindan uretilen jarlar (with-dependencies ve plain).
- `jvm/runtime/` ve `runtime.zip`: jlink tarafindan olusan trimmed JVM dagitimi.

## Sorun giderme ipuclari
- **LNK1104: cannot open agent_sut.exe** hatasi alirsaniz, exe muhtemelen calisiyordur. `Get-Process agent_sut` ile PID'yi bulun ve `Stop-Process` ile sonlandirip tekrar derleyin.
- Maven jar olusmuyor ise `JAVA_HOME` dogru JDK 21 dizinini gosteriyor mu kontrol edin.
- `/jvmstate` cagrilarinda `JVM_CAPTURE_FAILED` donuyorsa hedef proses 64-bit degil veya ajan enjekte edilemiyor olabilir; PID'in dogru oldugundan ve uygulamanin uygun haklara sahip oldugundan emin olun.
- DPI ile ilgili ekran yaklama sorunlarinda `EnableDpiAwareness` cagrilarinin basarili oldugunu dogrulamak icin output logunu kontrol edin.

Bu README, agent-sut gelistirme ve test surecinde gerekecek komutlarin buyuk cogunlugunu copy/paste ile kullanabileceginiz sekilde toplamayi hedefler. Yeni bir gelistirme ortami kurarken yukaridaki "Tum projeyi CMake ile derlemek" bolumunden baslayabilirsiniz.



