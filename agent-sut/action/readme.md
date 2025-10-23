# SUT Action -- API & Geliştirici Rehberi

Bu modül, lab ortamında çalışan **SUT-Action** ajanıdır. Windows'ta `SendInput` kullanarak fare/klavye eylemlerini uygular ve **ACK-only** bir yanıt üretir. **State döndürmez.** Controller, eylem sonrası güncel durumu görmek için ayrı `/state` endpoint'ini çağırmalıdır.

## İçindekiler

- [Klasör Yapısı](#klasör-yapısı)
- [Derleme & Çalıştırma](#derleme--çalıştırma)
- [HTTP API: `/action`](#http-api-action)
  - [İstek (Request)](#istek-request)
  - [Yanıt (Response)](#yanıt-response)
  - [Hata Kodları](#hata-kodları)
- [Desteklenen Action Tipleri](#desteklenen-action-tipleri)
  - [`click`](#click)
  - [`type`](#type)
  - [`key_combo`](#key_combo)
  - [`drag`](#drag)
  - [`scroll`](#scroll)
  - [`key_down` / `key_up`](#key_down--key_up)
  - [`move` / `hover`](#move--hover)
  - [`wait`](#wait)
- [Sık Kullanılan Senaryolar](#sık-kullanılan-senaryolar)
- [Tasarım İlkeleri](#tasarım-ilkeleri)
- [State ile Birleştirme](#state-ile-birleştirme)
- [İstemci İpuçları](#istemci-ipuçları)

## Klasör Yapısı

```
action/
├─ action_robot.h / .cpp     # Düşük seviye: SendInput ile input enjeksiyonu
├─ action_handler.h / .cpp   # Orta seviye: JSON → ActionRobot çağrıları
├─ main_action.cpp           # (test server) HTTP /action ve /healthz
```

### Dosyaların Rolü

- **action_robot**: Fare (Click/Drag/Scroll/Move) ve klavye (TypeText/KeyCombo/KeyDown/KeyUp) enjeksiyonu; Per-Monitor DPI v2
- **action_handler**: HTTP body (JSON) parse → step'leri sırayla uygular → **ACK-only JSON** döner; UTF-8 → UTF-16 dönüşüm (TypeText)
- **main_action**: Minimal HTTP servis (`/action`, `/healthz`) -- lokal test için


## HTTP API: `/action`

### İstek (Request)

```json
{
  "action_id": "string-opsiyonel-ama-önerilir",
  "coords_space": "physical",
  "steps": [
    /* Aşağıdaki action tiplerinden bir veya daha fazlası, sıralı */
  ]
}
```

**Batch yürütme**: `steps[]` sırayla uygulanır. Bir adım hata verirse yürütme durur; yanıtın `applied` alanı başarıyla tamamlanan adım sayısını gösterir.

### Yanıt (Response)

#### Başarılı
```json
{
  "status": "ok",
  "action_id": "...",
  "timestamp": 1737891234567,
  "applied": 3
}
```

#### Hata
```json
{
  "status": "error",
  "action_id": "...",
  "code": "INVALID_PAYLOAD|UNSUPPORTED_ACTION|TARGET_NOT_FOUND|INPUT_INJECTION_FAILED|ACCESS_DENIED|TIMEOUT",
  "detail": "insan okunur açıklama",
  "applied": 1,
  "timestamp": 1737891234567
}
```

### Hata Kodları

- **INVALID_PAYLOAD**: Zorunlu alan eksik/yanlış tip (örn. `steps[]` yok)
- **UNSUPPORTED_ACTION**: Tanınmayan `type`
- **TARGET_NOT_FOUND**: (İleride `element_id` desteği eklenirse) hedef çözülemedi
- **INPUT_INJECTION_FAILED**: `SendInput`/Win32 başarısız oldu
- **ACCESS_DENIED**: Farklı bütünlük seviyesi/UAC/secure desktop engeli
- **TIMEOUT**: Adım içi bekleme/iş süresi aşıldı (şu an opsiyonel)

## Desteklenen Action Tipleri

**Koordinatlar**: physical pixels (DPI-safe).  
**rect** verilirse merkezine tıklanır: `(x + w/2, y + h/2)`.

### `click`

```json
{
  "type": "click",
  "button": "left|right|middle",
  "click_count": 1,
  "modifiers": ["ctrl","alt","shift","win"],
  "target": {
    "rect": { "x": 150, "y": 150, "w": 100, "h": 100 }
    /* veya
    "point": { "x": 200, "y": 200 }
    */
  }
}
```

### `type`

```json
{
  "type": "type",
  "text": "Merhaba",
  "delay_ms": 10,
  "enter": true
}
```

### `key_combo`

```json
{
  "type": "key_combo",
  "combo": ["ctrl","shift","p"]
}
```

**Destekli örnek isimler**:  
`"ctrl"`, `"alt"`, `"shift"`, `"win"`, `"enter"`, `"esc"`, `"tab"`, `"space"`, `"backspace"`, `"delete"`, `"home"`, `"end"`, `"pgup"`, `"pgdn"`, `"left"`, `"right"`, `"up"`, `"down"`, `"f1"..."f12"` ve tek karakter tuşlar (`"a"`, `"5"`, `"."` vb).

### `drag`

```json
{
  "type": "drag",
  "from": { "x": 400, "y": 400 },
  "to":   { "x": 650, "y": 650 },
  "button": "left",
  "hold_ms": 120
}
```

### `scroll`

```json
{
  "type": "scroll",
  "delta": -240,
  "horizontal": false,
  "at": { "x": 800, "y": 600 }
}
```

### `key_down` / `key_up`

```json
{ "type": "key_down", "key": "shift" }
{ "type": "key_up",   "key": "shift"  }
```

### `move` / `hover`

```json
{
  "type": "move",
  "point": { "x": 200, "y": 200 },
  "settle_ms": 150
}
```

### `wait`

```json
{ "type": "wait", "ms": 300 }
```

## Sık Kullanılan Senaryolar

### Rect merkezine tıkla → yaz → Enter

```json
{
  "action_id": "rect_click_type",
  "steps": [
    {
      "type": "click",
      "button": "left",
      "target": { "rect": { "x": 150, "y": 150, "w": 100, "h": 100 } }
    },
    { "type": "type", "text": "Merhaba", "delay_ms": 10, "enter": true }
  ]
}
```

### Çift tık + Shift basılıyken

```json
{
  "action_id": "double_shift_click",
  "steps": [
    {
      "type": "click",
      "button": "left",
      "click_count": 2,
      "modifiers": ["shift"],
      "target": { "point": { "x": 1000, "y": 500 } }
    }
  ]
}
```

### Kısayol: Ctrl+S

```json
{
  "action_id": "save_combo",
  "steps": [ { "type": "key_combo", "combo": ["ctrl","s"] } ]
}
```

### Sürükle → Scroll

```json
{
  "action_id": "drag_then_scroll",
  "steps": [
    { "type": "drag", "from": { "x": 300, "y": 300 }, "to": { "x": 600, "y": 600 }, "button": "left", "hold_ms": 150 },
    { "type": "scroll", "delta": -240 }
  ]
}
```

### Odak al → yaz (Not Defteri örneği)

```json
{
  "action_id": "focus_and_type",
  "steps": [
    { "type": "move",  "point": { "x": 200, "y": 200 }, "settle_ms": 120 },
    { "type": "click", "button": "left", "target": { "point": { "x": 200, "y": 200 } } },
    { "type": "type",  "text": "Merhaba çğö 🙂", "delay_ms": 5, "enter": true }
  ]
}
```

## Tasarım İlkeleri

- **ACK-only**: `/action` state dönmez. Controller, adım sonrası `/state` çağırır
- **Physical pixels**: SUT prosesinde Per-Monitor DPI v2 açık; koordinatlar DPI'dan etkilenmez
- **Odak (focus)**: Klavye girişi odaktaki pencereye gider; gerekirse önce `click` ile odak alın
- **Güvenlik**: UAC/secure desktop'ta input engellenebilir; lab'da aynı bütünlük seviyesinde çalıştırın
- **Atomic batch**: Hata olduğunda yürütme durur; `applied`, tamamlanan adım sayısıdır


### PowerShell UTF-8 gönderimi:

```powershell
$utf8 = [System.Text.Encoding]::UTF8
Invoke-RestMethod -Uri http://127.0.0.1:18080/action `
  -Method POST `
  -Body ($utf8.GetBytes($body)) `
  -ContentType 'application/json; charset=utf-8'
```