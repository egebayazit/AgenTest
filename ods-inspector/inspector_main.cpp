// Third-party libraries MUST come BEFORE Windows.h (httplib uses winsock2)
#include "third_party/httplib.h"
#include "third_party/json.hpp"

// Standard library
#include <string>
#include <iostream>
#include <thread>
#include <iomanip>
#include <cstdlib>
#include <climits>
#include <vector>

// Windows headers (after httplib to avoid winsock conflicts)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <windowsx.h>

// UI Automation for WinDriver support
#include <UIAutomation.h>
#include <comdef.h>
#include <OleAuto.h>
#pragma comment(lib, "Uiautomationcore.lib")

// Local utilities
#include "win/capture.h"
#include "win/base64.h"

using json = nlohmann::json;
using namespace std;

// Configuration - ODS_HOST env var overrides default
std::string GetOdsHost() {
    const char* env = std::getenv("ODS_HOST");
    return env ? std::string(env) : "10.182.6.60";
}
const int ODS_API_PORT = 8000;
const std::string ENDPOINT_GET_ID = "/get-id-from-ods";
const std::string ENDPOINT_GET_COORDS = "/get-coords-from-ods";

HHOOK hKeyboardHook = NULL;
IUIAutomation* g_pAutomation = NULL;

// ============================================================================
// WIN ELEMENT STRUCT
// ============================================================================
struct WinElement {
    std::wstring name;
    std::wstring automationId;
    std::wstring value;
    bool found = false;
};

// ============================================================================
// WSTRING TO UTF8 HELPER
// ============================================================================
std::string WideToUtf8(const std::wstring& wstr) {
    if (wstr.empty()) return "";
    int size = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), nullptr, 0, nullptr, nullptr);
    if (size <= 0) return "";
    std::string result(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &result[0], size, nullptr, nullptr);
    return result;
}

// ============================================================================
// SUT-STYLE HELPERS (from agent-sut/win/uia_utils.cpp)
// ============================================================================
static std::wstring FromBSTR(BSTR b) { return b ? std::wstring(b, SysStringLen(b)) : L""; }
static inline bool HasPositiveArea(const RECT& r) { return (r.right > r.left) && (r.bottom > r.top); }

static inline RECT VirtualScreenRect() {
    RECT r{};
    r.left = GetSystemMetrics(SM_XVIRTUALSCREEN);
    r.top = GetSystemMetrics(SM_YVIRTUALSCREEN);
    r.right = r.left + GetSystemMetrics(SM_CXVIRTUALSCREEN);
    r.bottom = r.top + GetSystemMetrics(SM_CYVIRTUALSCREEN);
    return r;
}

static inline bool IntersectNonEmpty(const RECT& a, const RECT& b) {
    RECT out{};
    if (IntersectRect(&out, &a, &b)) return HasPositiveArea(out);
    return false;
}

static inline bool PointInRect(const RECT& r, int x, int y) {
    return x >= r.left && x < r.right && y >= r.top && y < r.bottom;
}

static inline long long RectArea(const RECT& r) {
    long long w = r.right - r.left;
    long long h = r.bottom - r.top;
    return (w > 0 && h > 0) ? w * h : LLONG_MAX;
}

// ============================================================================
// SUT-STYLE CACHE REQUEST (from agent-sut)
// ============================================================================
static IUIAutomationCacheRequest* CreateSmartCacheRequest(IUIAutomation* uia) {
    IUIAutomationCacheRequest* req = nullptr;
    if (FAILED(uia->CreateCacheRequest(&req))) return nullptr;

    req->AddProperty(UIA_NamePropertyId);
    req->AddProperty(UIA_ClassNamePropertyId);
    req->AddProperty(UIA_ControlTypePropertyId);
    req->AddProperty(UIA_BoundingRectanglePropertyId);
    req->AddProperty(UIA_AutomationIdPropertyId);
    req->AddProperty(UIA_IsEnabledPropertyId);
    req->AddProperty(UIA_IsOffscreenPropertyId);
    req->AddProperty(UIA_NativeWindowHandlePropertyId);
    req->AddProperty(UIA_ProcessIdPropertyId);

    // Light pattern checks
    req->AddPattern(UIA_InvokePatternId);
    req->AddPattern(UIA_ValuePatternId);
    req->AddPattern(UIA_SelectionItemPatternId);
    req->AddPattern(UIA_TogglePatternId);
    req->AddPattern(UIA_ExpandCollapsePatternId);
    req->AddPattern(UIA_ScrollPatternId);
    req->AddPattern(UIA_TextPatternId);
    req->AddPattern(UIA_LegacyIAccessiblePatternId);

    req->put_TreeScope(TreeScope_Element);
    return req;
}

static void GetSmartProp(IUIAutomationElement* e, PROPERTYID pid, VARIANT* v) {
    VariantInit(v);
    e->GetCachedPropertyValue(pid, v); 
}

// ============================================================================
// Extract element properties from CACHED element (SUT style)
// ============================================================================
static void ExtractCachedElementProps(IUIAutomationElement* elem, WinElement& result) {
    VARIANT v;
    
    // Get Name from cache
    GetSmartProp(elem, UIA_NamePropertyId, &v);
    if (v.vt == VT_BSTR && v.bstrVal) result.name = FromBSTR(v.bstrVal);
    VariantClear(&v);
    
    // Get AutomationId from cache
    GetSmartProp(elem, UIA_AutomationIdPropertyId, &v);
    if (v.vt == VT_BSTR && v.bstrVal) result.automationId = FromBSTR(v.bstrVal);
    VariantClear(&v);
    
    // Get Value - need to use pattern, can't cache this easily
    IUnknown* pUnk = nullptr;
    HRESULT hr = elem->GetCachedPattern(UIA_ValuePatternId, &pUnk);
    if (SUCCEEDED(hr) && pUnk) {
        IValueProvider* pValue = nullptr;
        hr = pUnk->QueryInterface(IID_PPV_ARGS(&pValue));
        if (SUCCEEDED(hr) && pValue) {
            BSTR bstrValue = nullptr;
            hr = pValue->get_Value(&bstrValue);
            if (SUCCEEDED(hr) && bstrValue) {
                result.value = FromBSTR(bstrValue);
                SysFreeString(bstrValue);
            }
            pValue->Release();
        }
        pUnk->Release();
    }
}

// ============================================================================
// GET WIN ELEMENT AT POINT - SUT-STYLE (from agent-sut)
// Uses CacheRequest + TreeWalker BFS exactly like snapshot_filtered
// ============================================================================
WinElement GetWinElementAtPoint(int x, int y) {
    WinElement result;
    
    if (!g_pAutomation) {
        return result;
    }
    
    const RECT vs = VirtualScreenRect();
    
    // Create cache request like SUT
    IUIAutomationCacheRequest* cacheReq = CreateSmartCacheRequest(g_pAutomation);
    if (!cacheReq) {
        return result;
    }
    
    // Get TreeWalker
    IUIAutomationTreeWalker* walker = nullptr;
    g_pAutomation->get_RawViewWalker(&walker);
    if (!walker) { 
        cacheReq->Release(); 
        return result; 
    }
    
    // Get window at mouse point (NOT foreground - inspector console becomes foreground on F4)
    // WindowFromPoint gives us the actual window under the cursor
    POINT mousePoint = { x, y };
    HWND hwndAtPoint = WindowFromPoint(mousePoint);
    HWND fgRoot = hwndAtPoint ? GetAncestor(hwndAtPoint, GA_ROOT) : nullptr;
    if (!fgRoot) fgRoot = hwndAtPoint;
    

    
    // Start from foreground window
    IUIAutomationElement* rootElem = nullptr;
    if (fgRoot) {
        g_pAutomation->ElementFromHandleBuildCache(fgRoot, cacheReq, &rootElem);
    }
    if (!rootElem) {
        g_pAutomation->GetRootElementBuildCache(cacheReq, &rootElem);
    }
    
    if (!rootElem) {
        walker->Release();
        cacheReq->Release();
        return result;
    }
    
    // BFS traversal exactly like SUT's snapshot_filtered
    struct QueueItem {
        IUIAutomationElement* elem;
        int depth;
    };
    std::vector<QueueItem> queue;
    queue.push_back({ rootElem, 0 });
    
    IUIAutomationElement* bestElement = nullptr;
    long long bestArea = LLONG_MAX;
    
    const int MAX_DEPTH = 18;  // Same as SUT
    const int MAX_ELEMENTS = 500;
    int processed = 0;
    int foundCount = 0;
    
    size_t head = 0;
    while (head < queue.size() && processed < MAX_ELEMENTS) {
        QueueItem item = queue[head++];
        IUIAutomationElement* elem = item.elem;
        processed++;
        
        VARIANT v;
        
        // Get control type for container detection
        GetSmartProp(elem, UIA_ControlTypePropertyId, &v);
        long cType = v.lVal;
        VariantClear(&v);
        
        // Get IsOffscreen
        GetSmartProp(elem, UIA_IsOffscreenPropertyId, &v);
        bool isOffscreen = (v.vt == VT_BOOL && v.boolVal == VARIANT_TRUE);
        VariantClear(&v);
        
        // Get BoundingRectangle
        RECT rect = {0,0,0,0};
        GetSmartProp(elem, UIA_BoundingRectanglePropertyId, &v);
        if ((v.vt & VT_ARRAY) && v.parray) {
            SAFEARRAY* sa = v.parray;
            double* p = nullptr; 
            SafeArrayAccessData(sa, (void**)&p);
            rect.left = (LONG)p[0]; 
            rect.top = (LONG)p[1];
            rect.right = (LONG)(p[0] + p[2]); 
            rect.bottom = (LONG)(p[1] + p[3]);
            SafeArrayUnaccessData(sa);
        }
        VariantClear(&v);
        
        bool posArea = HasPositiveArea(rect);
        bool onScreen = posArea && IntersectNonEmpty(rect, vs);
        bool containsPoint = posArea && PointInRect(rect, x, y);
        
        // --- IMPROVED PRUNING (same as SUT) ---
        bool skipSelf = false;
        bool isDialogOrWindow = (cType == UIA_WindowControlTypeId);
        
        // If offscreen, skip ONLY if not a Window which might contain visible popups
        if (item.depth > 0 && (!onScreen || isOffscreen) && !isDialogOrWindow) {
            skipSelf = true;
        }
        
        // Check if this element contains our target point (only if not skipped)
        if (!skipSelf && containsPoint) {
            foundCount++;
            long long area = RectArea(rect);
            
            // Get name to check if meaningful
            GetSmartProp(elem, UIA_NamePropertyId, &v);
            bool hasName = (v.vt == VT_BSTR && v.bstrVal && SysStringLen(v.bstrVal) > 0);
            VariantClear(&v);
            
            // Prefer smaller elements with names (more specific)
            if (hasName && area < bestArea) {
                if (bestElement) bestElement->Release();
                bestElement = elem;
                bestElement->AddRef();
                bestArea = area;
            } else if (!bestElement && area < bestArea) {
                bestElement = elem;
                bestElement->AddRef();
                bestArea = area;
            }
        }
        
        // Container detection for depth traversal (same as SUT)
        bool isContainer = (cType == UIA_WindowControlTypeId || cType == UIA_PaneControlTypeId || 
                            cType == UIA_GroupControlTypeId || cType == UIA_ListControlTypeId || 
                            cType == UIA_TableControlTypeId || cType == UIA_TreeControlTypeId ||
                            cType == UIA_DataGridControlTypeId || cType == UIA_CustomControlTypeId);
        
        // --- DEEP TRAVERSAL (same as SUT line 320) ---
        // Traverse if: not skipped AND depth < 18 AND (is container OR depth < 4)
        if (!skipSelf && item.depth < MAX_DEPTH && (isContainer || item.depth < 4)) {
            IUIAutomationElement* child = nullptr;
            if (SUCCEEDED(walker->GetFirstChildElementBuildCache(elem, cacheReq, &child)) && child) {
                int consecutiveOffscreen = 0;
                IUIAutomationElement* current = child;
                
                while (current) {
                    // Check child offscreen status
                    VARIANT vOff; 
                    VariantInit(&vOff);
                    current->GetCachedPropertyValue(UIA_IsOffscreenPropertyId, &vOff);
                    bool childOff = (vOff.vt == VT_BOOL && vOff.boolVal == VARIANT_TRUE);
                    VariantClear(&vOff);
                    
                    // Check child rect
                    VARIANT vR; 
                    VariantInit(&vR);
                    current->GetCachedPropertyValue(UIA_BoundingRectanglePropertyId, &vR);
                    RECT childRect = {0,0,0,0};
                    if ((vR.vt & VT_ARRAY) && vR.parray) {
                        SAFEARRAY* sa = vR.parray;
                        double* p = nullptr; 
                        SafeArrayAccessData(sa, (void**)&p);
                        childRect.left = (LONG)p[0]; 
                        childRect.top = (LONG)p[1];
                        childRect.right = (LONG)(p[0] + p[2]); 
                        childRect.bottom = (LONG)(p[1] + p[3]);
                        SafeArrayUnaccessData(sa);
                    }
                    VariantClear(&vR);
                    
                    bool childVisible = HasPositiveArea(childRect) && IntersectNonEmpty(childRect, vs);
                    bool childContainsPoint = HasPositiveArea(childRect) && PointInRect(childRect, x, y);
                    
                    if (childOff || !childVisible) {
                        consecutiveOffscreen++;
                    } else {
                        consecutiveOffscreen = 0;
                    }
                    
                    // Stop after 25 consecutive offscreen (same as SUT)
                    if (consecutiveOffscreen > 25) {
                        current->Release();
                        break;
                    }
                    
                    // Add ALL children to queue (same as SUT line 359)
                    queue.push_back({ current, item.depth + 1 });
                    
                    IUIAutomationElement* next = nullptr;
                    if (FAILED(walker->GetNextSiblingElementBuildCache(current, cacheReq, &next))) {
                        break;
                    }
                    current = next;
                }
            }
        }
        
        elem->Release();
    }
    

    
    // Clean up remaining queue
    for (size_t i = head; i < queue.size(); ++i) {
        if (queue[i].elem) queue[i].elem->Release();
    }
    
    walker->Release();
    cacheReq->Release();
    
    // Extract properties from best element
    if (bestElement) {
        result.found = true;
        ExtractCachedElementProps(bestElement, result);
        bestElement->Release();

    }
    
    return result;
}

// ============================================================================
// CLIPBOARD HELPER
// ============================================================================
std::string GetClipboardText() {
    if (!OpenClipboard(nullptr)) {
        return "";
    }

    HANDLE hData = GetClipboardData(CF_UNICODETEXT);
    if (hData == nullptr) {
        CloseClipboard();
        return "";
    }

    wchar_t* pszText = static_cast<wchar_t*>(GlobalLock(hData));
    if (pszText == nullptr) {
        CloseClipboard();
        return "";
    }

    // Convert UTF-16 to UTF-8
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, pszText, -1, NULL, 0, NULL, NULL);
    std::string strText(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, pszText, -1, &strText[0], size_needed, NULL, NULL);

    GlobalUnlock(hData);
    CloseClipboard();

    // Remove null terminator
    if (!strText.empty() && strText.back() == '\0') {
        strText.pop_back();
    }

    return strText;
}

// ============================================================================
// SCREENSHOT HELPER
// ============================================================================
std::string CaptureScreenToBase64() {
    std::vector<unsigned char> png_data;
    if (!CaptureScreenPng(png_data)) {
        return "";
    }
    return base64_encode(png_data.data(), png_data.size());
}

// ============================================================================
// F4/F3 HANDLER: Koordinattan Element Bulma
// removeHover: true = fareyi köşeye taşı (hover efektini kaldır)
// ============================================================================
void HandleF4Press(int mouse_x, int mouse_y, bool removeHover = false) {
    // Initialize COM for this thread (required for WIC)
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        std::cerr << "COM initialization failed in F4 thread!\n";
        return;
    }

    try {
        if (removeHover) {
            std::cout << "\n[F3 Pressed: Find Element (No Hover)]\n";
            // Fareyi ekranın sol üst köşesine taşı (hover'ı kaldır)
            SetCursorPos(0, 0);
            Sleep(100); // UI'ın hover state'i kaldırması için bekle
        } else {
            std::cout << "\n[F4 Pressed: Find Element by Coords]\n";
        }
        std::cout << "Coordinates: (" << mouse_x << ", " << mouse_y << ")\n";

        // ============================================
        // ODS Element Info
        // ============================================
        std::cout << "==========================================\n";
        std::cout << "[ODS]\n";
        
        std::string base64_image = CaptureScreenToBase64();
        
        // Fare pozisyonunu geri getir (removeHover modunda)
        if (removeHover) {
            SetCursorPos(mouse_x, mouse_y);
        }
        
        if (base64_image.empty()) {
            std::cout << "  [!] Screenshot capture failed\n";
        } else {
            json request_body;
            request_body["base64_image"] = base64_image;
            request_body["x"] = mouse_x;
            request_body["y"] = mouse_y;

            httplib::Client client(GetOdsHost(), ODS_API_PORT);
            client.set_read_timeout(30, 0);  // Reduced timeout
            client.set_write_timeout(10, 0);
            client.set_connection_timeout(5, 0);

            auto res = client.Post(ENDPOINT_GET_ID.c_str(), 
                                   request_body.dump(), 
                                   "application/json");

            if (!res) {
                std::cout << "  [!] ODS server not reachable\n";
            } else if (res->status != 200) {
                std::cout << "  [!] ODS error: HTTP " << res->status << "\n";
            } else {
                json response = json::parse(res->body);
                std::string status = response["status"];

                if (status == "success") {
                    std::string content_name = response["content_name"];
                    int element_id = response["element_id"];
                    
                    std::cout << "  name : \"" << content_name << "\"\n";
                    std::cout << "  id   : " << element_id << "\n";
                } else {
                    std::string message = response.value("message", "No element found");
                    std::cout << "  [!] " << message << "\n";
                }
            }
        }

        // ============================================
        // WINDRIVER (UI Automation) Element Info
        // ============================================
        std::cout << "==========================================\n";
        std::cout << "[WINDRIVER]\n";
        WinElement winElem = GetWinElementAtPoint(mouse_x, mouse_y);
        if (winElem.found) {
            std::cout << "  name  : \"" << WideToUtf8(winElem.name) << "\"\n";
            std::cout << "  id    : \"" << WideToUtf8(winElem.automationId) << "\"\n";
            std::cout << "  value : \"" << WideToUtf8(winElem.value) << "\"\n";
        } else {
            std::cout << "  [!] No Win element found at this point\n";
        }
        std::cout << "==========================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    CoUninitialize();
}


// ============================================================================
// F5 HANDLER: Element İsminden Koordinat Bulma
// ============================================================================
void HandleF5Press() {
    // Initialize COM for this thread (required for WIC)
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        std::cerr << "COM initialization failed in F5 thread!\n";
        return;
    }

    try {
        std::cout << "\n[F5 Pressed: Find Coords by Element Name]\n";

        // 1. Clipboard'dan text al
        std::string clipboard_text = GetClipboardText();
        if (clipboard_text.empty()) {
            std::cerr << "Clipboard is empty! Copy an element name first.\n";
            CoUninitialize();
            return;
        }

        std::cout << "Clipboard Content: \"" << clipboard_text << "\"\n";
        std::cout << "Capturing screenshot...\n";

        // 2. Screenshot al
        std::string base64_image = CaptureScreenToBase64();
        if (base64_image.empty()) {
            std::cerr << "Screenshot capture failed!\n";
            CoUninitialize();
            return;
        }

        std::cout << "Calling " << ENDPOINT_GET_COORDS << "...\n";

        // 3. JSON request oluştur
        json request_body;
        request_body["base64_image"] = base64_image;
        request_body["content_id"] = clipboard_text;

        // 4. HTTP POST request
        httplib::Client client(GetOdsHost(), ODS_API_PORT);
        client.set_read_timeout(120, 0);  // 120 seconds
        client.set_write_timeout(30, 0);
        client.set_keep_alive(true);

        auto res = client.Post(ENDPOINT_GET_COORDS.c_str(), 
                               request_body.dump(), 
                               "application/json");


        // 5. Response kontrolü
        if (!res) {
            std::cerr << "API Error: No response received\n";
            CoUninitialize();
            return;
        }


        if (res->status != 200) {
            std::cerr << "API Error: HTTP " << res->status << "\n";
            CoUninitialize();
            return;
        }


        // 6. JSON response parse et
        json response = json::parse(res->body);
        std::string status = response["status"];

        std::cout << "==========================================\n";

        if (status == "success") {
            int count = response["count"];
            auto matches = response["matches"];

            std::cout << "Element Search Results - Matches: " << count << "\n";

            int index = 1;
            for (const auto& match : matches) {
                std::string content = match["content"];
                int element_id = match["id"];
                int x = match["x"];
                int y = match["y"];
                double match_score = match["match_score"];

                std::cout << "  --- Match [" << index << "] ---\n";
                std::cout << "    Content Name : \"" << content << "\"\n";
                std::cout << "    Element ID   : " << element_id << "\n";
                std::cout << "    Coordinates  : (" << x << ", " << y << ")\n";
                std::cout << "    Match Score  : " << std::fixed << std::setprecision(4) << match_score;
                
                if (match_score >= 1.0) {
                    std::cout << " (Exact Match)\n";
                } else {
                    std::cout << " (Fuzzy Match)\n";
                }

                index++;
            }
        } else {
            std::string message = response.value("message", "No elements found");
            std::cout << "\n[!] " << message << "\n";
        }

        std::cout << "==========================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    CoUninitialize();
}

// ============================================================================
// KEYBOARD HOOK CALLBACK
// ============================================================================
LRESULT CALLBACK KeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode != HC_ACTION || wParam != WM_KEYDOWN) {
        return CallNextHookEx(hKeyboardHook, nCode, wParam, lParam);
    }

    KBDLLHOOKSTRUCT* pKeyboard = (KBDLLHOOKSTRUCT*)lParam;

    switch (pKeyboard->vkCode) {
        case VK_F3: {
            // F3: Hover efekti olmadan inspect (fare köşeye taşınır)
            std::thread f3Thread([]() {
                POINT pt;
                GetCursorPos(&pt);
                HandleF4Press(pt.x, pt.y, true);  // removeHover = true
            });
            f3Thread.detach();
            break;
        }

        case VK_F4: {
            // F4: Normal inspect (hover dahil)
            std::thread f4Thread([]() {
                POINT pt;
                GetCursorPos(&pt);
                HandleF4Press(pt.x, pt.y, false);  // removeHover = false
            });
            f4Thread.detach();
            break;
        }

        case VK_F5: {
            // F5 tuşuna basıldığında F5 işleyicisini ayrı bir thread'de başlat
            std::thread f5Thread([]() {
                HandleF5Press();
            });
            f5Thread.detach();
            break;
        }
    }

    return CallNextHookEx(hKeyboardHook, nCode, wParam, lParam);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    // Set DPI awareness FIRST - before any Windows API calls
    // This ensures GetCursorPos returns physical coordinates
    SetProcessDPIAware();

    // Initialize COM for WIC (Windows Imaging Component) used by screenshot capture
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM!\n";
        return 1;
    }

    // Initialize UI Automation for WinDriver support
    hr = CoCreateInstance(CLSID_CUIAutomation, NULL, CLSCTX_INPROC_SERVER, 
                          IID_IUIAutomation, (void**)&g_pAutomation);
    if (FAILED(hr) || !g_pAutomation) {
        std::cerr << "Warning: Failed to initialize UI Automation (WinDriver will be disabled)\n";
    }

    // Console encoding (Set to UTF-8 for better compatibility)
    SetConsoleOutputCP(CP_UTF8);
    
    // Tüm loglar std::cout ile yapılıyor
    std::cout << "=== ODS + WINDRIVER INSPECTOR ===\n";
    std::cout << "F3: Find element (no hover)\n";
    std::cout << "F4: Find element (with hover)\n";
    std::cout << "F5: Find coordinates by name (clipboard)\n";
    std::cout << "ODS Server: http://" << GetOdsHost() << ":" << ODS_API_PORT << "\n";
    std::cout << "WinDriver: " << (g_pAutomation ? "Active" : "Disabled") << "\n";
    std::cout << "=================================\n\n";

    // Install keyboard hook
    hKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardProc, NULL, 0);
    if (!hKeyboardHook) {
        std::cerr << "Failed to install keyboard hook!\n";
        if (g_pAutomation) g_pAutomation->Release();
        CoUninitialize();
        return 1;
    }

    std::cout << "Inspector is running. Press Ctrl+C to exit.\n\n";

    // Message loop
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Cleanup
    if (hKeyboardHook) {
        UnhookWindowsHookEx(hKeyboardHook);
    }
    if (g_pAutomation) {
        g_pAutomation->Release();
        g_pAutomation = NULL;
    }

    CoUninitialize();
    return 0;
}