// Third-party libraries MUST come BEFORE Windows.h (httplib uses winsock2)
#include "third_party/httplib.h"
#include "third_party/json.hpp"

// Standard library
#include <string>
#include <iostream>
#include <thread>
#include <iomanip>

// Windows headers (after httplib to avoid winsock conflicts)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <windowsx.h>

// Local utilities
#include "win/capture.h"
#include "win/base64.h"

using json = nlohmann::json;
using namespace std;

// Configuration
const std::string ODS_API_HOST = "10.182.6.60";
const int ODS_API_PORT = 8000;
const std::string ENDPOINT_GET_ID = "/get-id-from-ods";
const std::string ENDPOINT_GET_COORDS = "/get-coords-from-ods";

HHOOK hKeyboardHook = NULL;

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
// F4 HANDLER: Koordinattan Element Bulma
// ============================================================================
void HandleF4Press(int mouse_x, int mouse_y) {
    // Initialize COM for this thread (required for WIC)
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        std::cerr << "COM initialization failed in F4 thread!\n";
        return;
    }

    try {
        std::cout << "\n[F4 Pressed: Find Element by Coords]\n";
        std::cout << "Capturing screenshot...\n";

        // 1. Screenshot al
        std::string base64_image = CaptureScreenToBase64();
        if (base64_image.empty()) {
            std::cerr << "Screenshot capture failed!\n";
            CoUninitialize();
            return;
        }

        // Parse PNG dimensions from base64 (Unused, but kept for completeness)
        int screenshot_width = 1920;   // Default
        int screenshot_height = 1200;  // Default
        
        if (base64_image.length() >= 44) {
            std::string header_b64 = base64_image.substr(0, 44);
            std::string decoded = base64_decode(header_b64);
            if (decoded.length() >= 24) {
                screenshot_width = ((unsigned char)decoded[16] << 24) | 
                                  ((unsigned char)decoded[17] << 16) |
                                  ((unsigned char)decoded[18] << 8) |
                                  ((unsigned char)decoded[19]);
                screenshot_height = ((unsigned char)decoded[20] << 24) |
                                   ((unsigned char)decoded[21] << 16) |
                                   ((unsigned char)decoded[22] << 8) |
                                   ((unsigned char)decoded[23]);
            }
        }

        std::cout << "Calling " << ENDPOINT_GET_ID << "...\n";

        // 2. JSON request oluştur
        json request_body;
        request_body["base64_image"] = base64_image;
        request_body["x"] = mouse_x;  // Direct coordinates
        request_body["y"] = mouse_y;  // Direct coordinates

        // 3. HTTP POST request
        httplib::Client client(ODS_API_HOST, ODS_API_PORT);
        client.set_read_timeout(120, 0);  // 120 seconds
        client.set_write_timeout(30, 0);  // 30 seconds
        client.set_keep_alive(true);

        auto res = client.Post(ENDPOINT_GET_ID.c_str(), 
                               request_body.dump(), 
                               "application/json");


        // 4. Response kontrolü
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


        // 5. JSON response parse et
        json response = json::parse(res->body);
        std::string status = response["status"];

        std::cout << "==========================================\n";

        if (status == "success") {
            std::string content_name = response["content_name"];
            int element_id = response["element_id"];
            
            std::cout << "Element Found:\n";
            std::cout << "  Content Name : \"" << content_name << "\"\n";
            std::cout << "  Element ID   : " << element_id << "\n";
            std::cout << "  Coordinates  : (" << mouse_x << ", " << mouse_y << ")\n";
        } else {
            std::string message = response.value("message", "No element found");
            std::cout << "\n[!] " << message << "\n";
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
        httplib::Client client(ODS_API_HOST, ODS_API_PORT);
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
        case VK_F4: {
            // F4 tuşuna basıldığında fare koordinatlarını al ve F4 işleyicisini ayrı bir thread'de başlat
            std::thread f4Thread([]() {
                POINT pt;
                GetCursorPos(&pt);
                HandleF4Press(pt.x, pt.y);
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

    // Console encoding (Set to UTF-8 for better compatibility)
    SetConsoleOutputCP(CP_UTF8);
    
    // Tüm loglar std::cout ile yapılıyor
    std::cout << "=== ODS INSPECTOR ===\n";
    std::cout << "F4: Find element at cursor\n";
    std::cout << "F5: Find coordinates by name (from clipboard)\n";
    std::cout << "Server: http://" << ODS_API_HOST << ":" << ODS_API_PORT << "\n";
    std::cout << "====================\n\n";

    // Install keyboard hook
    hKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardProc, NULL, 0);
    if (!hKeyboardHook) {
        std::cerr << "Failed to install keyboard hook!\n";
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

    CoUninitialize();
    return 0;
}