// Third-party libraries MUST come BEFORE Windows.h (httplib uses winsock2)
#include "third_party/httplib.h"
#include "third_party/json.hpp"

// Standard library
#include <string>
#include <iostream>
#include <thread>
#include <iomanip>
#include <cstdlib>

// Windows headers (after httplib to avoid winsock conflicts)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <windowsx.h>

// UI Automation for WinDriver support
#include <UIAutomation.h>
#include <comdef.h>
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
// GET WIN ELEMENT AT POINT (UI Automation)
// ============================================================================
WinElement GetWinElementAtPoint(int x, int y) {
    WinElement result;
    
    if (!g_pAutomation) {
        return result;
    }
    
    POINT pt = { x, y };
    IUIAutomationElement* pElement = nullptr;
    
    HRESULT hr = g_pAutomation->ElementFromPoint(pt, &pElement);
    if (FAILED(hr) || !pElement) {
        return result;
    }
    
    result.found = true;
    
    // Get Name
    BSTR bstrName = nullptr;
    hr = pElement->get_CurrentName(&bstrName);
    if (SUCCEEDED(hr) && bstrName) {
        result.name = std::wstring(bstrName, SysStringLen(bstrName));
        SysFreeString(bstrName);
    }
    
    // Get AutomationId
    BSTR bstrId = nullptr;
    hr = pElement->get_CurrentAutomationId(&bstrId);
    if (SUCCEEDED(hr) && bstrId) {
        result.automationId = std::wstring(bstrId, SysStringLen(bstrId));
        SysFreeString(bstrId);
    }
    
    // Get Value (from IValueProvider pattern)
    IValueProvider* pValueProvider = nullptr;
    hr = pElement->GetCurrentPattern(UIA_ValuePatternId, (IUnknown**)&pValueProvider);
    if (SUCCEEDED(hr) && pValueProvider) {
        BSTR bstrValue = nullptr;
        hr = pValueProvider->get_Value(&bstrValue);
        if (SUCCEEDED(hr) && bstrValue) {
            result.value = std::wstring(bstrValue, SysStringLen(bstrValue));
            SysFreeString(bstrValue);
        }
        pValueProvider->Release();
    }
    
    pElement->Release();
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