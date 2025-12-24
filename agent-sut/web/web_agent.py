import sys
import json
import asyncio
import base64
from playwright.async_api import async_playwright

# Hedef tarayıcı debug portu
CDP_URL = "http://127.0.0.1:9222"

async def get_web_state():
    try:
        async with async_playwright() as p:
            try:
                browser = await p.chromium.connect_over_cdp(CDP_URL)
            except Exception as e:
                print(json.dumps({"error": f"Could not connect to browser at {CDP_URL}", "detail": str(e)}))
                return

            if not browser.contexts:
                print(json.dumps({"error": "No browser context found"}))
                return
            
            context = browser.contexts[0]
            if not context.pages:
                print(json.dumps({"error": "No pages found"}))
                return

            page = context.pages[0] 
            
            # Screenshot al
            try:
                screenshot_bytes = await page.screenshot(type="png")
                b64_screenshot = base64.b64encode(screenshot_bytes).decode("utf-8")
            except:
                b64_screenshot = ""

            # 2. Elementleri Topla
            # BURASI DEĞİŞTİ: Viewport kontrolü eklendi
            js_script = """
            () => {
                const results = [];
                const allElements = document.querySelectorAll('*');
                
                // Anlık Görünen Ekran Boyutları (Viewport)
                const vh = window.innerHeight || document.documentElement.clientHeight;
                const vw = window.innerWidth || document.documentElement.clientWidth;

                // Meta veriler
                const meta = {
                    outerWidth: window.outerWidth,
                    outerHeight: window.outerHeight,
                    innerWidth: window.innerWidth,
                    innerHeight: window.innerHeight,
                    screenX: window.screenX,
                    screenY: window.screenY
                };

                for (let el of allElements) {
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;

                    const isClickable = (
                        el.tagName === 'BUTTON' || 
                        el.tagName === 'A' || 
                        el.tagName === 'INPUT' || 
                        el.tagName === 'TEXTAREA' ||
                        el.tagName === 'SELECT' ||
                        el.onclick != null ||
                        el.getAttribute('role') === 'button'
                    );

                    if (!isClickable) continue;

                    const rect = el.getBoundingClientRect();
                    
                    if (rect.width <= 0 || rect.height <= 0) continue;

                    // --- VIEWPORT KONTROLÜ (KRİTİK KISIM) ---
                    // Element şu anki görüş alanında mı?
                    const inViewport = (
                        rect.top < vh &&     // Alt sınırın üstünde mi?
                        rect.bottom > 0 &&   // Üst sınırın altında mı?
                        rect.left < vw &&    // Sağ sınırın solunda mı?
                        rect.right > 0       // Sol sınırın sağında mı?
                    );

                    // Eğer görünür alanda değilse (scroll'da aşağıda veya yukarıda kaldıysa) atla.
                    if (!inViewport) continue;
                    // ----------------------------------------

                    results.push({
                        tagName: el.tagName,
                        id: el.id,
                        name: el.getAttribute('name') || el.innerText || el.getAttribute('aria-label') || '',
                        type: el.getAttribute('type') || '',
                        rect: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        }
                    });
                }
                return { elements: results, meta: meta };
            }
            """
            
            dom_data = await page.evaluate(js_script)

            output = {
                "stateType": "web",
                "url": page.url,
                "title": await page.title(),
                "b64": b64_screenshot,
                "meta": dom_data["meta"],
                "elements": []
            }

            idx = 0
            for el in dom_data["elements"]:
                # İsimlerdeki yeni satırları temizle ki JSON tek satır kalsın
                clean_name = el["name"].replace("\n", " ").strip()[:200]
                
                output["elements"].append({
                    "idx": idx,
                    "id": el["id"],
                    "name": clean_name, 
                    "controlType": el["tagName"], 
                    "rect": {
                        "x": el["rect"]["x"],
                        "y": el["rect"]["y"],
                        "w": el["rect"]["width"],
                        "h": el["rect"]["height"]
                    },
                    "center": {
                        "x": el["rect"]["x"] + el["rect"]["width"]/2,
                        "y": el["rect"]["y"] + el["rect"]["height"]/2
                    }
                })
                idx += 1

            print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"error": "Unexpected error in web agent", "detail": str(e)}))

if __name__ == "__main__":
    asyncio.run(get_web_state())