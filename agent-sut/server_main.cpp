// server_main.cpp
#include "third_party/httplib.h"
#include "third_party/json.hpp"
#include "jvm/jvm_bridge.h"

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
//#include "base64.h"

using json = nlohmann::json;



// ========================= DPI awareness =========================
static void EnableDpiAwareness() {
    if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
        SetProcessDPIAware();
    }
}

// ========================= küçük yardımcılar ======================
static std::string ws2utf8(const std::wstring& ws){
    if(ws.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
    if(len<=0) return {};
    std::string out((size_t)len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), &out[0], len, nullptr, nullptr);
    return out;
}

static std::wstring ExeDir(){
    wchar_t buf[MAX_PATH]{}; GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring p(buf); size_t pos = p.find_last_of(L"\\/");
    return (pos==std::wstring::npos)? L"." : p.substr(0,pos);
}
static void EnsureDir(const std::wstring& path){
    CreateDirectoryW(path.c_str(), nullptr); // varsa NO-OP
}
static bool WriteAllBytes(const std::wstring& path, const void* data, DWORD size){
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if(h==INVALID_HANDLE_VALUE) return false;
    DWORD written=0; BOOL ok = WriteFile(h, data, size, &written, nullptr);
    CloseHandle(h); return ok && (written==size);
}
static bool WriteAllTextUtf8(const std::wstring& path, const std::string& utf8){
    return WriteAllBytes(path, utf8.data(), (DWORD)utf8.size());
}
static std::wstring TimeStamp(){
    SYSTEMTIME st; GetLocalTime(&st);
    wchar_t buf[64];
    swprintf(buf, 64, L"%04u%02u%02u_%02u%02u%02u_%03u",
             st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return buf;
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

    auto elems = uia.snapshot_filtered(128);  // daha kucuk & filtreli
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

        const std::wstring base = ExeDir();
        const std::wstring ts   = TimeStamp();
        const std::wstring statesDir = base + L"\\winstate_logs";
        const std::wstring shotsDir  = base + L"\\winstate_screenshots";
        EnsureDir(statesDir);
        EnsureDir(shotsDir);

        const std::wstring jsonPath = statesDir + L"\\winstate_" + ts + L".json";
        const std::wstring pngPath  = shotsDir  + L"\\winstate_" + ts + L".png";

        const std::string jsonStr = j.dump();
        WriteAllTextUtf8(jsonPath, jsonStr);
        if(!png_raw.empty()){
            WriteAllBytes(pngPath, png_raw.data(), (DWORD)png_raw.size());
        }

        res.set_content(jsonStr, "application/json");
    });


    // ---- Combined STATE endpoint (tries JVM state first, falls back to Windows state)
    svr.Post("/state", [&](const httplib::Request& req, httplib::Response& res){
        std::string jvmError;
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
            const std::wstring base = ExeDir();
            const std::wstring ts   = TimeStamp();
            const std::wstring logsDir = base + L"\\state_logs";
            EnsureDir(logsDir);
            const std::wstring jsonPath = logsDir + L"\\state_jvm_" + ts + L".json";
            const std::wstring shotsDir = base + L"\\state_screenshots";
            EnsureDir(shotsDir);
            const std::wstring pngPath = shotsDir + L"\\state_jvm_" + ts + L".png";

            bool screenshotWritten = false;
            if (b64Data && !b64Data->empty()) {
                response["b64"] = *b64Data;
                response["screenshot"] = { {"format", "png"} };

                std::string decoded = base64_decode(*b64Data);
                if (!decoded.empty()) {
                    WriteAllBytes(pngPath, decoded.data(), static_cast<DWORD>(decoded.size()));
                    screenshotWritten = true;
                }
            } else {
                response["screenshot"] = { {"format", "png"} };
            }

            const std::string responseStr = response.dump();

            json logResponse = response;
            if (screenshotWritten) {
                logResponse["screenshot"]["filePath"] = ws2utf8(pngPath);
            } else {
                logResponse["screenshot"]["filePath"] = nullptr;
            }
            WriteAllTextUtf8(jsonPath, logResponse.dump());

            res.set_content(responseStr, "application/json");
            return;
        } catch (const std::exception& ex) {
            jvmError = ex.what();
        }

        std::vector<unsigned char> png_raw;
        auto j = make_state(uia, &png_raw, /*want_screenshot=*/true);
        j["stateType"] = "windows";
        if (!jvmError.empty()) {
            j["fallbackReason"] = jvmError;
        }

        const std::wstring base = ExeDir();
        const std::wstring ts   = TimeStamp();
        const std::wstring logsDir = base + L"\\state_logs";
        const std::wstring shotsDir  = base + L"\\state_screenshots";
        EnsureDir(logsDir);
        EnsureDir(shotsDir);

        const std::wstring jsonPath = logsDir + L"\\state_windows_" + ts + L".json";
        const std::wstring pngPath  = shotsDir  + L"\\state_windows_" + ts + L".png";

        const std::string jsonStr = j.dump();
        WriteAllTextUtf8(jsonPath, jsonStr);
        if(!png_raw.empty()){
            WriteAllBytes(pngPath, png_raw.data(), (DWORD)png_raw.size());
        }

        res.set_content(jsonStr, "application/json");
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
            const std::wstring base = ExeDir();
            const std::wstring ts   = TimeStamp();
            const std::wstring logsDir = base + L"\\jvmstate_logs";
            EnsureDir(logsDir);
            const std::wstring errPath = logsDir + L"\\jvmstate_error_" + ts + L".json";
            WriteAllTextUtf8(errPath, err.dump());

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
            const std::wstring base = ExeDir();
            const std::wstring ts   = TimeStamp();
            const std::wstring logsDir = base + L"\\jvmstate_logs";
            const std::wstring shotsDir = base + L"\\jvmstate_screenshots";
            EnsureDir(logsDir);
            EnsureDir(shotsDir);
            const std::wstring jsonPath = logsDir + L"\\jvmstate_" + ts + L".json";
            const std::wstring pngPath = shotsDir + L"\\jvmstate_" + ts + L".png";
            bool screenshotWritten = false;
            if (b64Data && !b64Data->empty()) {
                response["b64"] = *b64Data;
                response["screenshot"] = { {"format", "png"} };

                std::string decoded = base64_decode(*b64Data);
                if (!decoded.empty()) {
                    WriteAllBytes(pngPath, decoded.data(), static_cast<DWORD>(decoded.size()));
                    screenshotWritten = true;
                }
            } else {
                response["screenshot"] = { {"format", "png"} };
            }

            const std::string responseStr = response.dump();

            json logResponse = response;
            if (screenshotWritten) {
                logResponse["screenshot"]["filePath"] = ws2utf8(pngPath);
            } else {
                logResponse["screenshot"]["filePath"] = nullptr;
            }
            WriteAllTextUtf8(jsonPath, logResponse.dump());

            res.set_content(responseStr, "application/json");
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
            const std::wstring base = ExeDir();
            const std::wstring ts   = TimeStamp();
            const std::wstring logsDir = base + L"\\jvmstate_logs";
            EnsureDir(logsDir);
            const std::wstring errPath = logsDir + L"\\jvmstate_error_parse_" + ts + L".json";
            WriteAllTextUtf8(errPath, err.dump());

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
