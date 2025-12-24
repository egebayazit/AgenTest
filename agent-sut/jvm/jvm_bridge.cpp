#include "jvm_bridge.h"

#include <Windows.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::mutex& BridgeMutex() {
    static std::mutex m;
    return m;
}

std::wstring GetExeDir() {
    wchar_t buffer[MAX_PATH]{};
    DWORD len = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    if (len == 0 || len == MAX_PATH) {
        return L".";
    }
    std::wstring path(buffer, len);
    size_t pos = path.find_last_of(L"\\/");
    if (pos == std::wstring::npos) {
        return L".";
    }
    return path.substr(0, pos);
}

std::string WideToUtf8(const std::wstring& ws) {
    if (ws.empty()) {
        return {};
    }
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0) {
        return {};
    }
    std::string result(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()), result.data(), len, nullptr, nullptr);
    return result;
}

std::string ReadFileUtf8(const fs::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return {};
    }
    std::string content;
    ifs.seekg(0, std::ios::end);
    content.resize(static_cast<size_t>(ifs.tellg()));
    ifs.seekg(0, std::ios::beg);
    ifs.read(content.data(), static_cast<std::streamsize>(content.size()));
    return content;
}

std::optional<fs::path> FindLatestSnapshotJson(const fs::path& logDir) {
    if (!fs::exists(logDir) || !fs::is_directory(logDir)) {
        return std::nullopt;
    }
    std::optional<fs::directory_entry> latest;
    for (const auto& entry : fs::directory_iterator(logDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto& p = entry.path();
        if (p.extension() != ".json") {
            continue;
        }
        const auto filename = p.filename().wstring();
        if (filename.rfind(L"snapshot-", 0) != 0) {
            continue;
        }
        if (!latest || entry.last_write_time() > latest->last_write_time()) {
            latest = entry;
        }
    }
    if (latest) {
        return latest->path();
    }
    return std::nullopt;
}

std::wstring EscapeForPowerShell(const fs::path& path) {
    std::wstring source = path.wstring();
    std::wstring escaped;
    escaped.reserve(source.size());
    for (wchar_t ch : source) {
        if (ch == L'\'') {
            escaped += L"''";
        } else {
            escaped.push_back(ch);
        }
    }
    return escaped;
}

std::optional<fs::file_time_type> GetLastWriteTime(const fs::path& path) {
    std::error_code ec;
    auto time = fs::last_write_time(path, ec);
    if (ec) {
        return std::nullopt;
    }
    return time;
}

struct EmbeddedBlob {
    const BYTE* data = nullptr;
    DWORD size = 0;
};

std::optional<EmbeddedBlob> LoadEmbeddedJar() {
    HRSRC res = FindResourceW(nullptr, L"JVM_AGENT", RT_RCDATA);
    if (!res) {
        return std::nullopt;
    }
    HGLOBAL handle = LoadResource(nullptr, res);
    if (!handle) {
        return std::nullopt;
    }
    const BYTE* data = static_cast<const BYTE*>(LockResource(handle));
    if (!data) {
        return std::nullopt;
    }
    DWORD size = SizeofResource(nullptr, res);
    if (size == 0) {
        return std::nullopt;
    }
    return EmbeddedBlob{data, size};
}

std::string FormatWindowsError(const char* context) {
    DWORD code = GetLastError();
    LPWSTR buffer = nullptr;
    DWORD size = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPWSTR>(&buffer),
        0,
        nullptr);

    std::wstring message;
    if (size && buffer) {
        message.assign(buffer, size);
        LocalFree(buffer);
    }
    std::string utf8Message = WideToUtf8(message);
    if (!utf8Message.empty() && utf8Message.back() == '\n') {
        while (!utf8Message.empty() && (utf8Message.back() == '\n' || utf8Message.back() == '\r')) {
            utf8Message.pop_back();
        }
    }
    std::string formatted = std::string(context) + " (error " + std::to_string(code) + ")";
    if (!utf8Message.empty()) {
        formatted += ": " + utf8Message;
    }
    return formatted;
}

bool EnsureJarAvailable(const fs::path& jarPath) {
    auto embeddedOpt = LoadEmbeddedJar();
    if (!embeddedOpt) {
        return fs::exists(jarPath);
    }
    const EmbeddedBlob& blob = *embeddedOpt;

    std::error_code ec;
    fs::create_directories(jarPath.parent_path(), ec);

    bool needsWrite = true;
    if (fs::exists(jarPath)) {
        auto size = fs::file_size(jarPath, ec);
        if (!ec && static_cast<uintmax_t>(blob.size) == size) {
            needsWrite = false;
        }
    }
    if (!needsWrite) {
        return true;
    }

    std::ofstream ofs(jarPath, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(blob.data), static_cast<std::streamsize>(blob.size));
    return ofs.good();
}

std::optional<EmbeddedBlob> LoadEmbeddedRuntime() {
    HRSRC res = FindResourceW(nullptr, L"JVM_RUNTIME", RT_RCDATA);
    if (!res) {
        return std::nullopt;
    }
    HGLOBAL handle = LoadResource(nullptr, res);
    if (!handle) {
        return std::nullopt;
    }
    const BYTE* data = static_cast<const BYTE*>(LockResource(handle));
    if (!data) {
        return std::nullopt;
    }
    DWORD size = SizeofResource(nullptr, res);
    if (size == 0) {
        return std::nullopt;
    }
    return EmbeddedBlob{data, size};
}

bool RunCommand(const std::wstring& command, const std::wstring& workingDir, DWORD& exitCode, std::string& error) {
    STARTUPINFOW si{};
    si.cb = sizeof(STARTUPINFOW);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> cmdBuffer(command.begin(), command.end());
    cmdBuffer.push_back(L'\0');

    const wchar_t* workDirPtr = workingDir.empty() ? nullptr : workingDir.c_str();

    BOOL created = CreateProcessW(
        nullptr,
        cmdBuffer.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NO_WINDOW,
        nullptr,
        workDirPtr,
        &si,
        &pi);

    if (!created) {
        exitCode = static_cast<DWORD>(-1);
        error = FormatWindowsError("Failed to launch process");
        return false;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD localExit = 0;
    GetExitCodeProcess(pi.hProcess, &localExit);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    exitCode = localExit;
    return exitCode == 0;
}

bool EnsureRuntimeAvailable(const fs::path& runtimeDir) {
    const fs::path javaExe = runtimeDir / L"bin" / L"java.exe";
    if (fs::exists(javaExe)) {
        return true;
    }

    auto embedded = LoadEmbeddedRuntime();
    if (!embedded) {
        return false;
    }

    std::error_code ec;
    fs::create_directories(runtimeDir.parent_path(), ec);

    fs::path zipPath = runtimeDir.parent_path() / L"runtime.zip";
    {
        std::ofstream ofs(zipPath, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            return false;
        }
        ofs.write(reinterpret_cast<const char*>(embedded->data), static_cast<std::streamsize>(embedded->size));
        if (!ofs.good()) {
            return false;
        }
    }

    std::wstring command = L"powershell.exe -NoProfile -Command \"Try { Expand-Archive -Force -LiteralPath '"
        + EscapeForPowerShell(zipPath) + L"' -DestinationPath '" + EscapeForPowerShell(runtimeDir.parent_path())
        + L"' } Catch { exit 1 }\"";

    DWORD exitCode = 0;
    std::string error;
    if (!RunCommand(command, runtimeDir.parent_path().wstring(), exitCode, error)) {
        return false;
    }

    std::error_code removeEc;
    fs::remove(zipPath, removeEc);

    return fs::exists(javaExe);
}

std::wstring BuildCommandLine(const fs::path& runtimeDir, const fs::path& jarPath, const fs::path& logDir, const std::optional<DWORD>& pid) {
    fs::path javaExe = runtimeDir / L"bin" / L"java.exe";
    fs::path logPath = logDir / L"java-run.log";
    std::wstring command = L"cmd.exe /c \"\"" + javaExe.wstring() + L"\" -jar \"" + jarPath.wstring() + L"\"";
    if (pid) {
        command += L" " + std::to_wstring(*pid);
    }
    // Pass log directory as second argument so Java agent knows where to write snapshot files
    command += L" \"" + logDir.wstring() + L"\"";
    command += L" > \"" + logPath.wstring() + L"\" 2>&1\"";
    return command;
}

void CleanupLogDirectory(const fs::path& logDir) {
    std::error_code ec;
    if (!fs::exists(logDir)) {
        return;
    }
    for (const auto& entry : fs::directory_iterator(logDir)) {
        if (entry.is_regular_file()) {
            fs::remove(entry.path(), ec);
        } else if (entry.is_directory()) {
            fs::remove_all(entry.path(), ec);
        }
    }
}

} // namespace

namespace jvm_bridge {

namespace {

fs::path ResolveJvmDirectory(const fs::path& exeDir) {
    return exeDir / L"jvm";
}

std::optional<DWORD> GetActiveWindowPid() {
    HWND foreground = GetForegroundWindow();
    if (!foreground) {
        return std::nullopt;
    }
    HWND root = GetAncestor(foreground, GA_ROOTOWNER);
    HWND target = root ? root : foreground;
    DWORD pid = 0;
    GetWindowThreadProcessId(target, &pid);
    if (pid == 0) {
        return std::nullopt;
    }
    return pid;
}

} // namespace

SnapshotResult CaptureSnapshot(std::optional<unsigned long> requestedPid) {
    std::lock_guard<std::mutex> lock(BridgeMutex());
    SnapshotResult result;

    const fs::path exeDir = fs::path(GetExeDir());
    const fs::path jvmDir = ResolveJvmDirectory(exeDir);
    const fs::path jarPath = jvmDir / L"target" / L"jvm-element-finder-1.0-SNAPSHOT-jar-with-dependencies.jar";
    const fs::path runtimeDir = jvmDir / L"runtime";
    const fs::path logDir = jvmDir / L"logs";

    if (!EnsureJarAvailable(jarPath)) {
        result.errorMessage = "Agent jar not found at " + WideToUtf8(jarPath.wstring());
        return result;
    }

    if (!EnsureRuntimeAvailable(runtimeDir)) {
        result.errorMessage = "Failed to prepare embedded Java runtime.";
        CleanupLogDirectory(logDir);
        return result;
    }

    if (!fs::exists(logDir)) {
        std::error_code ec;
        fs::create_directories(logDir, ec);
        if (ec) {
            result.errorMessage = "Failed to create log directory: " + WideToUtf8(logDir.wstring());
            return result;
        }
    }

    auto previousLatest = FindLatestSnapshotJson(logDir);
    std::optional<fs::file_time_type> previousTime;
    if (previousLatest) {
        previousTime = GetLastWriteTime(*previousLatest);
    }

    std::optional<DWORD> activePid;
    if (requestedPid) {
        activePid = static_cast<DWORD>(*requestedPid);
    } else {
        activePid = GetActiveWindowPid();
    }

    fs::path logFile = logDir / L"java-run.log";
    std::wstring command = BuildCommandLine(runtimeDir, jarPath, logDir, activePid);
    DWORD exitCode = 0;
    std::string launchError;
    if (!RunCommand(command, jvmDir.wstring(), exitCode, launchError)) {
        result.exitCode = static_cast<int>(exitCode);
        if (activePid) {
            result.usedPid = static_cast<unsigned long>(*activePid);
        }
        if (!launchError.empty()) {
            result.errorMessage = launchError;
        } else {
            std::string logContent = ReadFileUtf8(logFile);
            if (!logContent.empty()) {
                if (logContent.size() > 4000) {
                    logContent.resize(4000);
                    logContent.append("...\n");
                }
                result.errorMessage = "Java process exited with code " + std::to_string(exitCode) + "\n" + logContent;
            } else {
                result.errorMessage = "Java process exited with code " + std::to_string(exitCode);
            }
        }
        CleanupLogDirectory(logDir);
        return result;
    }

    result.exitCode = static_cast<int>(exitCode);
    if (activePid) {
        result.usedPid = static_cast<unsigned long>(*activePid);
    }

    std::optional<fs::path> latestJson = FindLatestSnapshotJson(logDir);
    auto latestTime = latestJson ? GetLastWriteTime(*latestJson) : std::optional<fs::file_time_type>{};

    if (!latestJson) {
        result.errorMessage = "No snapshot json produced.";
        CleanupLogDirectory(logDir);
        return result;
    }
    if (previousLatest && latestJson->wstring() == previousLatest->wstring()) {
        if (!latestTime || !previousTime || *latestTime <= *previousTime) {
            result.errorMessage = "Snapshot json not updated.";
            CleanupLogDirectory(logDir);
            return result;
        }
    }

    if (!latestTime) {
        result.errorMessage = "No new snapshot json produced.";
        CleanupLogDirectory(logDir);
        return result;
    }

    const fs::path pngPath = latestJson->parent_path() / latestJson->stem().wstring().append(L".png");

    if (!fs::exists(pngPath)) {
        result.errorMessage = "Snapshot png not found: " + WideToUtf8(pngPath.wstring());
        CleanupLogDirectory(logDir);
        return result;
    }

    result.snapshotJson = ReadFileUtf8(*latestJson);
    result.snapshotJsonPath = latestJson->wstring();
    result.screenshotPath = pngPath.wstring();

    if (result.snapshotJson.empty()) {
        result.errorMessage = "Snapshot json is empty.";
        CleanupLogDirectory(logDir);
        return result;
    }

    result.success = true;
    CleanupLogDirectory(logDir);
    return result;
}

} // namespace jvm_bridge

