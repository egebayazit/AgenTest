#pragma once

#include <optional>
#include <string>

namespace jvm_bridge {

struct SnapshotResult {
    bool success = false;
    std::string snapshotJson;
    std::wstring snapshotJsonPath;
    std::wstring screenshotPath;
    std::string errorMessage;
    int exitCode = 0;
    std::optional<unsigned long> usedPid;
};

SnapshotResult CaptureSnapshot(std::optional<unsigned long> requestedPid);

} // namespace jvm_bridge
