// main.cpp
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#define NOMINMAX

#include "../third_party/httplib.h"
#include "../third_party/json.hpp"

#include <Windows.h>
#include <ShellScalingAPI.h>   // DPI awareness
#pragma comment(lib, "Shcore.lib")

#include <string>
#include <vector>

#include "uia_utils.h"
#include "capture.h"
#include "base64.h"

using json = nlohmann::json;

// ---------- DPI awareness ----------
static void EnableDpiAwareness() {
    // En modern yol; başarısız olursa geri uyumlu API'lere düş
    if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
        SetProcessDPIAware();
    }
}

// ---------- küçük yardımcılar ----------
static std::string ws2utf8(const std::wstring& ws){
    if(ws.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
    if(len<=0) return {};
    std::string out((size_t)len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), &out[0], len, nullptr, nullptr);
    return out;
}

static std::wstring ExeDir(){
    wchar_t buf[MAX_PATH]{};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring p(buf);
    size_t pos = p.find_last_of(L"\\/");
    return (pos==std::wstring::npos)? L"." : p.substr(0,pos);
}

static void EnsureDir(const std::wstring& path){
    CreateDirectoryW(path.c_str(), nullptr); // varsa NO-OP
}

static bool WriteAllBytes(const std::wstring& path, const void* data, DWORD size){
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if(h==INVALID_HANDLE_VALUE) return false;
    DWORD written=0;
    BOOL ok = WriteFile(h, data, size, &written, nullptr);
    CloseHandle(h);
    return ok && (written==size);
}

static bool WriteAllTextUtf8(const std::wstring& path, const std::string& utf8){
    return WriteAllBytes(path, utf8.data(), (DWORD)utf8.size());
}

static std::wstring TimeStamp(){
    SYSTEMTIME st;
    GetLocalTime(&st);
    wchar_t buf[64];
    swprintf(buf, 64, L"%04u%02u%02u_%02u%02u%02u_%03u",
             st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return buf;
}

// ---------- patterns & path JSON ----------
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

// ---------- Eleman JSON (lite) ----------
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

// make_state: hem JSON üretir hem de (varsa) PNG baytlarını döndürür
static json make_state(UiaSession& uia, std::vector<unsigned char>* raw_png_out, bool want_screenshot = true){
    json out;
    FillScreenInfo(out);

    auto elems = uia.snapshot_filtered(128);  // daha küçük & filtreli
    out["elements"] = json::array();
    HWND fgRoot = GetAncestor(GetForegroundWindow(), GA_ROOT);
    int idx=0;
    for(const auto& e : elems){
        out["elements"].push_back(ElemToJsonLite(e, idx++, fgRoot));
    }

    if(want_screenshot){
        std::vector<unsigned char> png;
        if(CaptureScreenPng(png)){
            out["screenshot"] = { {"format","png"}, {"b64", base64_encode(png.data(), png.size())} };
            if(raw_png_out) *raw_png_out = std::move(png);
        } else {
            out["screenshot"] = { {"format","png"}, {"b64",""} };
            if(raw_png_out) raw_png_out->clear();
        }
    }

    out["timestamp"] = (uint64_t) GetTickCount64();
    return out;
}

// ---------- main ----------
int main(){
    EnableDpiAwareness();    // <- kırpılma için kritik

    UiaSession uia;
    httplib::Server svr;

    svr.Get("/ping", [](auto&, auto& res){
        res.set_content("pong", "text/plain");
    });

    svr.Post("/state", [&](const httplib::Request&, httplib::Response& res){
        std::vector<unsigned char> png_raw;
        auto j = make_state(uia, &png_raw, /*want_screenshot=*/true);

        // ---- Diske yaz: state_logs + screenshots ----
        const std::wstring base = ExeDir();
        const std::wstring ts = TimeStamp();
        const std::wstring statesDir = base + L"\\state_logs";
        const std::wstring shotsDir  = base + L"\\screenshots";
        EnsureDir(statesDir);
        EnsureDir(shotsDir);

        const std::wstring jsonPath = statesDir + L"\\state_" + ts + L".json";
        const std::wstring pngPath  = shotsDir  + L"\\screen_" + ts + L".png";

        // JSON (pretty değil -> daha küçük)
        const std::string jsonStr = j.dump();
        WriteAllTextUtf8(jsonPath, jsonStr);

        // PNG (ham veriden)
        if(!png_raw.empty()){
            WriteAllBytes(pngPath, png_raw.data(), (DWORD)png_raw.size());
        }

        // HTTP yanıtı
        res.set_content(jsonStr, "application/json");
    });

    printf("SUT /state server on http://0.0.0.0:8765\n");
    svr.listen("0.0.0.0", 8765);
    return 0;
}
