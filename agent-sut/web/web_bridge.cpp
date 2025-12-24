#include "web_bridge.h"
#include "../win/resource.h" // Resource ID'leri için
#include <Windows.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>

namespace fs = std::filesystem;

namespace {

// Geçici dizin yolunu al (Örn: C:\Users\X\AppData\Local\Temp\)
std::wstring GetTempDirectory() {
    wchar_t buffer[MAX_PATH];
    GetTempPathW(MAX_PATH, buffer);
    return std::wstring(buffer);
}

// Gömülü EXE'yi Temp klasörüne çıkarır
bool ExtractEmbeddedAgent(const std::wstring& outputPath) {
    HMODULE hModule = GetModuleHandle(NULL);
    HRSRC hRes = FindResource(hModule, MAKEINTRESOURCE(IDR_WEB_AGENT_EXE), RT_RCDATA);
    if (!hRes) return false;

    HGLOBAL hMem = LoadResource(hModule, hRes);
    if (!hMem) return false;
    DWORD size = SizeofResource(hModule, hRes);
    void* data = LockResource(hMem);

    HANDLE hFile = CreateFileW(outputPath.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return false;

    DWORD written;
    BOOL result = WriteFile(hFile, data, size, &written, NULL);
    CloseHandle(hFile);

    return (result && written == size);
}

// Komut çalıştırma ve çıktı alma
bool RunCommandCaptureOutput(const std::wstring& cmd, std::string& output) {
    SECURITY_ATTRIBUTES sa; sa.nLength = sizeof(SECURITY_ATTRIBUTES); 
    sa.bInheritHandle = TRUE; sa.lpSecurityDescriptor = NULL;

    HANDLE hRead, hWrite;
    if (!CreatePipe(&hRead, &hWrite, &sa, 0)) return false;
    SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si = { sizeof(STARTUPINFOW) };
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdOutput = hWrite; si.hStdError = hWrite; si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi = { 0 };
    std::vector<wchar_t> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back(0);

    if (!CreateProcessW(NULL, cmdBuf.data(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(hWrite); CloseHandle(hRead); return false;
    }
    CloseHandle(hWrite);

    char buffer[4096];
    DWORD bytesRead;
    while (ReadFile(hRead, buffer, sizeof(buffer), &bytesRead, NULL) && bytesRead > 0) {
        output.append(buffer, bytesRead);
    }
    WaitForSingleObject(pi.hProcess, 15000); 
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(hRead);
    return true;
}

} // namespace

namespace web_bridge {

WebSnapshotResult CaptureWebSnapshot() {
    WebSnapshotResult result;
    
    std::wstring tempExePath = GetTempDirectory() + L"agent_web_worker.exe";

    // Resource'tan çıkar
    if (!ExtractEmbeddedAgent(tempExePath)) {
        if (!fs::exists(tempExePath)) {
            // DÜZELTME BURADA: fs::path(...).string() kullanarak wide->string dönüşümünü güvenli yapıyoruz.
            result.errorMessage = "Failed to extract embedded web agent to: " + fs::path(tempExePath).string();
            return result;
        }
    }

    std::wstring cmd = L"\"" + tempExePath + L"\"";
    
    if (RunCommandCaptureOutput(cmd, result.jsonOutput)) {
        if (result.jsonOutput.find("{") != std::string::npos) {
            result.success = true;
        } else {
            result.errorMessage = "Invalid output from web agent: " + result.jsonOutput;
        }
    } else {
        result.errorMessage = "Failed to run extracted web agent.";
    }

    return result;
}

} // namespace web_bridge