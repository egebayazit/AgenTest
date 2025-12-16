// capture.cpp
#define WIN32_LEAN_AND_MEAN
#include "capture.h"
#include <Windows.h>
#include <wincodec.h>

#pragma comment(lib, "Windowscodecs.lib")

// HBITMAP -> PNG (WIC ile)
static bool SaveHBITMAPToPNG(HBITMAP hbmp, std::vector<unsigned char>& out) {
    IWICImagingFactory* fac = nullptr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&fac)))) return false;

    IWICBitmap* wicBmp = nullptr;
    HRESULT hr = fac->CreateBitmapFromHBITMAP(hbmp, nullptr, WICBitmapUseAlpha, &wicBmp);
    if (FAILED(hr)) { fac->Release(); return false; }

    IStream* memStream = nullptr;
    CreateStreamOnHGlobal(NULL, TRUE, &memStream);

    IWICStream* stream = nullptr;
    fac->CreateStream(&stream);
    stream->InitializeFromIStream(memStream);

    IWICBitmapEncoder* enc = nullptr;
    fac->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc);
    enc->Initialize(stream, WICBitmapEncoderNoCache);

    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* pb = nullptr;
    enc->CreateNewFrame(&frame, &pb);
    frame->Initialize(pb);

    UINT w = 0, h = 0;
    wicBmp->GetSize(&w, &h);
    frame->SetSize(w, h);

    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    frame->SetPixelFormat(&fmt);

    std::vector<BYTE> buf(w * h * 4);
    WICRect rc{ 0, 0, (INT)w, (INT)h };
    wicBmp->CopyPixels(&rc, w * 4, (UINT)buf.size(), buf.data());
    frame->WritePixels(h, w * 4, (UINT)buf.size(), buf.data());

    frame->Commit();
    enc->Commit();

    // memory stream'i vektöre kopyala
    STATSTG st{};
    memStream->Stat(&st, STATFLAG_NONAME);
    ULONG sz = (ULONG)st.cbSize.QuadPart;
    out.resize(sz);

    LARGE_INTEGER zero{};
    memStream->Seek(zero, STREAM_SEEK_SET, nullptr);
    ULONG read = 0;
    memStream->Read(out.data(), sz, &read);

    if (pb) pb->Release();
    frame->Release();
    enc->Release();
    stream->Release();
    memStream->Release();
    wicBmp->Release();
    fac->Release();
    return !out.empty();
}

bool CaptureScreenPng(std::vector<unsigned char>& out_png) {
    // Get PHYSICAL screen resolution (not DPI-scaled)
    DEVMODE dm;
    ZeroMemory(&dm, sizeof(dm));
    dm.dmSize = sizeof(dm);
    
    int width = 1920;   // Default
    int height = 1200;  // Default
    
    if (EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm)) {
        width = dm.dmPelsWidth;
        height = dm.dmPelsHeight;
    } else {
        // Fallback to primary monitor
        POINT origin{ 0, 0 };
        HMONITOR hmon = MonitorFromPoint(origin, MONITOR_DEFAULTTOPRIMARY);
        MONITORINFOEXW mi{};
        mi.cbSize = sizeof(mi);
        if (GetMonitorInfoW(hmon, &mi)) {
            const RECT& r = mi.rcMonitor;
            width = r.right - r.left;
            height = r.bottom - r.top;
        }
    }

    HDC hdc = GetDC(NULL);
    if (!hdc) return false;

    HDC mem = CreateCompatibleDC(hdc);
    if (!mem) { ReleaseDC(NULL, hdc); return false; }

    HBITMAP bmp = CreateCompatibleBitmap(hdc, width, height);
    if (!bmp) { DeleteDC(mem); ReleaseDC(NULL, hdc); return false; }

    HGDIOBJ old = SelectObject(mem, bmp);
    // CAPTUREBLT: capture layered windows too
    BitBlt(mem, 0, 0, width, height, hdc, 0, 0, SRCCOPY | CAPTUREBLT);
    SelectObject(mem, old);

    DeleteDC(mem);
    ReleaseDC(NULL, hdc);

    bool ok = SaveHBITMAPToPNG(bmp, out_png);
    DeleteObject(bmp);
    return ok;
}

bool GetScreenDimensions(int& width, int& height) {
    POINT origin{ 0, 0 };
    HMONITOR hmon = MonitorFromPoint(origin, MONITOR_DEFAULTTOPRIMARY);

    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(hmon, &mi)) return false;

    const RECT& r = mi.rcMonitor;
    width = r.right - r.left;
    height = r.bottom - r.top;
    return true;
}