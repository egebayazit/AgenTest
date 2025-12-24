// server_main.cpp
#include "third_party/httplib.h"
#include "third_party/json.hpp"
#include "jvm/jvm_bridge.h"
#include "web/web_bridge.h" // <--- Web Bridge Header

#include <Windows.h>
#include <ShellScalingAPI.h>   // DPI awareness
#pragma comment(lib, "Shcore.lib")

#include <string>
#include <optional>
#include <stdexcept>
#include "win/base64.h"

#include <vector>
#include <iostream>

// ---- ACTION tarafı ----
#include "action/action_handler.h"

// ---- WİN tarafı ----
#include "win/uia_utils.h"
#include "win/capture.h"

using json = nlohmann::json;

// ========================= DPI awareness =========================
static void EnableDpiAwareness() {
    if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
        SetProcessDPIAware();
    }
}

// ========================= Web Helper Fonksiyonları =========================
static bool IsBrowserWindow(HWND hwnd) {
    wchar_t className[256];
    if (GetClassNameW(hwnd, className, 256) > 0) {
        std::wstring name(className);
        // DEBUG LOG
        std::wcout << L"[DEBUG] Pencere Sinif Adi: " << name << std::endl;
        
        // Esnek arama (Contains mantığı)
        if (name.find(L"Chrome_WidgetWin") != std::wstring::npos) return true;
        if (name.find(L"Mozilla") != std::wstring::npos) return true;
        if (name.find(L"Edge") != std::wstring::npos) return true;
    }
    return false;
}

static POINT GetBrowserViewportOffset(HWND hwnd) {
    POINT pt = {0, 0};
    RECT clientRect;
    if (GetClientRect(hwnd, &clientRect)) {
        pt.x = clientRect.left;
        pt.y = clientRect.top;
        ClientToScreen(hwnd, &pt);
    }
    return pt;
}
// ====================================================================================

// ========================= küçük yardımcılar ======================
static std::string ws2utf8(const std::wstring& ws){
    if(ws.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
    if(len<=0) return {};
    std::string out((size_t)len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), &out[0], len, nullptr, nullptr);
    return out;
}

// ========================= winstate JSON yardımcıları =================
static json PatternsToJson(const UiaPatterns& p){
    return {
        {"invoke", p.invoke}, {"value", p.value}, {"selectionItem", p.selectionItem},
        {"toggle", p.toggle}, {"expandCollapse", p.expandCollapse},
        {"scroll", p.scroll}, {"text", p.text}
    };
}
static json PathToJsonLite(const std::vector<std::wstring>& path){
    json a = json::array();
    int n = (int)path.size();
    for(int i = std::max(0, n-3); i<n; ++i) a.push_back(ws2utf8(path[i]));
    return a;
}
static json ElemToJsonLite(const UiaElem& e, int idx, HWND fgRoot) {
    const int w = e.rect.right - e.rect.left;
    const int h = e.rect.bottom - e.rect.top;
    const int cx = e.rect.left + (w/2);
    const int cy = e.rect.top + (h/2);

    bool windowActive = false;
    if(e.hwnd){
        HWND root = GetAncestor(e.hwnd, GA_ROOT);
        windowActive = (root == fgRoot);
    }

    json j = {
        {"idx", idx},
        {"name", ws2utf8(e.name)},
        {"controlType", ws2utf8(e.controlType)},
        {"rect", { {"l", e.rect.left}, {"t", e.rect.top}, {"r", e.rect.right}, {"b", e.rect.bottom} }},
        {"center", {{"x", cx}, {"y", cy}}},
        {"enabled", e.enabled},
        {"windowActive",windowActive},
        {"patterns", PatternsToJson(e.patterns)},
        {"path", PathToJsonLite(e.path)}
    };
    return j;
}
static void FillScreenInfo(json& out){
    out["screen"]["w"] = UiaSession::ScreenW();
    out["screen"]["h"] = UiaSession::ScreenH();
    HDC hdc = GetDC(NULL);
    int dpiX = GetDeviceCaps(hdc, LOGPIXELSX);
    int dpiY = GetDeviceCaps(hdc, LOGPIXELSY);
    ReleaseDC(NULL, hdc);
    out["screen"]["dpiX"] = dpiX;
    out["screen"]["dpiY"] = dpiY;
}
static json make_state(UiaSession& uia, std::vector<unsigned char>* raw_png_out, bool want_screenshot = true){
    json out;
    FillScreenInfo(out);

    auto elems = uia.snapshot_filtered(500);  // daha kucuk & filtreli
    out["elements"] = json::array();
    HWND fgRoot = GetAncestor(GetForegroundWindow(), GA_ROOT);
    int idx = 0;
    for (const auto& e : elems) {
        out["elements"].push_back(ElemToJsonLite(e, idx++, fgRoot));
    }

    if (want_screenshot) {
        std::vector<unsigned char> png;
        if (CaptureScreenPng(png)) {
            const std::string b64 = base64_encode(reinterpret_cast<const unsigned char*>(png.data()), png.size());
            out["screenshot"] = {
                {"format", "png"},
                {"b64", b64}
            };
            if (raw_png_out) *raw_png_out = std::move(png);
        } else {
            out["screenshot"] = {
                {"format", "png"},
                {"b64", ""}
            };
            if (raw_png_out) raw_png_out->clear();
        }
    }

    out["timestamp"] = static_cast<uint64_t>(GetTickCount64());
    return out;
}

// ========================= main ======================
int main(){
    EnableDpiAwareness(); // <- kırpılma için kritik
    httplib::Server svr;

    // ACTION handler (mevcut action main'inden)
    action::ActionHandler action_handler;

    // STATE context (UIA oturumu)
    UiaSession uia;

    // ---- sağlık kontrolü
    svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res){
        res.set_content(R"({"status":"ok"})","application/json");
    });

    // ---- ACTION endpoint (ACK-only)
    svr.Post("/action", [&action_handler](const httplib::Request& req, httplib::Response& res){
        const auto ct = req.get_header_value("Content-Type");
        if(ct.find("application/json")==std::string::npos){
            res.status=400;
            res.set_content(R"({"status":"error","code":"INVALID_CONTENT_TYPE"})","application/json");
            return;
        }
        res.set_content(action_handler.Handle(req.body), "application/json");
    });

    // ---- WINDOWS STATE endpoint (JSON + screenshot + element listesi)
    svr.Post("/winstate", [&](const httplib::Request&, httplib::Response& res){
        std::vector<unsigned char> png_raw;
        auto j = make_state(uia, &png_raw, /*want_screenshot=*/true);
        j["stateType"] = "windows";
        res.set_content(j.dump(), "application/json");
    });


    // ---- Combined STATE endpoint (tries WEB -> JVM -> Windows)
    svr.Post("/state", [&](const httplib::Request& req, httplib::Response& res){
        
        // ======================= Web Modülü Entegrasyonu Başlangıcı =======================
        HWND fg = GetForegroundWindow();
        bool isBrowser = IsBrowserWindow(fg);
        std::string webError;

        // 1. WEB MODÜLÜ (Eğer tarayıcı ise)
        if (isBrowser) {
            std::cout << "[INFO] Browser tespit edildi. Web ajani calistiriliyor..." << std::endl;
            auto webRes = web_bridge::CaptureWebSnapshot();
            if (webRes.success) {
                try {
                    // --- JSON Temizliği ---
                    std::string cleanJson = webRes.jsonOutput;
                    size_t jsonStart = cleanJson.find('{'); // İlk süslü parantez
                    size_t jsonEnd = cleanJson.rfind('}');  // Son süslü parantez
                    
                    if (jsonStart != std::string::npos && jsonEnd != std::string::npos) {
                        // Node.js uyarılarını at, sadece { ... } arasını al
                        cleanJson = cleanJson.substr(jsonStart, jsonEnd - jsonStart + 1);
                        
                        json webJson = json::parse(cleanJson); // Artık temiz
                        
                        if (!webJson.contains("error")) {
                            // Koordinat düzeltme
                            POINT vpOffset = GetBrowserViewportOffset(fg);
                            int offsetX = vpOffset.x;
                            int offsetY = vpOffset.y;

                            if (webJson.contains("elements")) {
                                for (auto& el : webJson["elements"]) {
                                    if (el.contains("rect")) {
                                        el["rect"]["l"] = (int)(el["rect"]["x"].get<double>() + offsetX);
                                        el["rect"]["t"] = (int)(el["rect"]["y"].get<double>() + offsetY);
                                        el["rect"]["r"] = (int)(el["rect"]["l"].get<int>() + el["rect"]["w"].get<double>());
                                        el["rect"]["b"] = (int)(el["rect"]["t"].get<int>() + el["rect"]["h"].get<double>());
                                    }
                                    if (el.contains("center")) {
                                        el["center"]["x"] = (int)(el["center"]["x"].get<double>() + offsetX);
                                        el["center"]["y"] = (int)(el["center"]["y"].get<double>() + offsetY);
                                    }
                                }
                            }

                            res.set_content(webJson.dump(), "application/json");
                            return; // ÇIKIŞ
                        } else {
                            webError = webJson["error"].get<std::string>();
                            std::cout << "[ERROR] Web JSON Error: " << webError << std::endl;
                        }
                    } else {
                        webError = "Invalid JSON format (No brackets found). Raw: " + webRes.jsonOutput;
                        std::cout << "[ERROR] " << webError << std::endl;
                    }

                } catch (std::exception& e) {
                    webError = std::string(e.what()) + " | Raw: " + webRes.jsonOutput;
                    std::cout << "[ERROR] Web Parse Error: " << webError << std::endl;
                }
            } else {
                webError = webRes.errorMessage;
                std::cout << "[ERROR] Web Bridge Error: " << webError << std::endl;
            }
        }
        // ======================= Web Modülü Entegrasyonu Bitişi =======================


        std::string jvmError;
        // Eğer Browser ise JVM denemeye gerek yok, direkt Windows'a düşsün (Performans için)
        if (!isBrowser) {
            try {
                auto snapshot = jvm_bridge::CaptureSnapshot(std::nullopt);
                if (!snapshot.success) {
                    throw std::runtime_error(snapshot.errorMessage);
                }
                json payload = json::parse(snapshot.snapshotJson);
                json response = json::object();
                response["stateType"] = "jvm";
                std::optional<std::string> b64Data;
                for (auto it = payload.begin(); it != payload.end(); ++it) {
                    if (it.key() == "b64" && it.value().is_string()) {
                        b64Data = it.value().get<std::string>();
                        continue;
                    }
                    response[it.key()] = it.value();
                }

                if (b64Data && !b64Data->empty()) {
                    response["b64"] = *b64Data;
                    response["screenshot"] = { {"format", "png"} };
                } else {
                    response["screenshot"] = { {"format", "png"} };
                }

                res.set_content(response.dump(), "application/json");
                return;
            } catch (const std::exception& ex) {
                jvmError = ex.what();
            }
        }

        std::vector<unsigned char> png_raw;
        auto j = make_state(uia, &png_raw, /*want_screenshot=*/true);
        j["stateType"] = "windows";
        if (!jvmError.empty()) {
            j["fallbackReason"] = jvmError;
        }
        // Web hatası varsa onu da fallback mesajına ekle
        if (!webError.empty()) {
             if (j.contains("fallbackReason")) {
                 std::string existing = j["fallbackReason"];
                 j["fallbackReason"] = "[Web: " + webError + "] " + existing;
             } else {
                 j["fallbackReason"] = "[Web: " + webError + "]";
             }
        }

        res.set_content(j.dump(), "application/json");
    });

    // ---- JVM STATE endpoint
    svr.Post("/jvmstate", [&](const httplib::Request& req, httplib::Response& res){
        std::optional<unsigned long> pidOverride;
        if (auto pidParam = req.get_param_value("pid"); !pidParam.empty()) {
            try {
                pidOverride = static_cast<unsigned long>(std::stoul(pidParam));
            } catch (...) {
                res.status = 400;
                res.set_content(R"({"status":"error","code":"INVALID_PID","message":"pid query parameter must be numeric"})", "application/json");
                return;
            }
        } else if (!req.body.empty()) {
            try {
                auto bodyJson = json::parse(req.body);
                if (bodyJson.contains("pid") && !bodyJson["pid"].is_null()) {
                    pidOverride = bodyJson["pid"].get<unsigned long>();
                }
            } catch (const std::exception& ex) {
                res.status = 400;
                json err = {
                    {"status", "error"},
                    {"code", "INVALID_JSON"},
                    {"message", ex.what()}
                };
                res.set_content(err.dump(), "application/json");
                return;
            }
        }

        auto snapshot = jvm_bridge::CaptureSnapshot(pidOverride);
        if (!snapshot.success) {
            json err = {
                {"status", "error"},
                {"code", "JVM_CAPTURE_FAILED"},
                {"message", snapshot.errorMessage},
                {"exitCode", snapshot.exitCode},
                {"stateType", "jvm"}
            };
            if (snapshot.usedPid) {
                err["pid"] = *snapshot.usedPid;
            }
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }

        try {
            json payload = json::parse(snapshot.snapshotJson);
            json response = json::object();
            response["stateType"] = "jvm";
            std::optional<std::string> b64Data;
            for (auto it = payload.begin(); it != payload.end(); ++it) {
                if (it.key() == "b64" && it.value().is_string()) {
                    b64Data = it.value().get<std::string>();
                    continue;
                }
                response[it.key()] = it.value();
            }
            if (b64Data && !b64Data->empty()) {
                response["b64"] = *b64Data;
                response["screenshot"] = { {"format", "png"} };
            } else {
                response["screenshot"] = { {"format", "png"} };
            }
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& ex) {
            json err = {
                {"status", "error"},
                {"code", "JVM_RESPONSE_PARSE_FAILED"},
                {"message", ex.what()},
                {"stateType", "jvm"}
            };
            if (snapshot.usedPid) {
                err["pid"] = *snapshot.usedPid;
            }
            res.status = 500;
            res.set_content(err.dump(), "application/json");
        }
    });

    const char* host = "0.0.0.0";
    const int   port = 18080;

    std::cout << "SUT server listening on http://" << host << ":" << port
              << " (endpoints: /healthz, /action, /winstate, /state, /jvmstate)\n";

    if(!svr.listen(host, port)){
        std::cerr << "Failed to bind on port " << port << "\n";
        return 1;
    }
    return 0;
}