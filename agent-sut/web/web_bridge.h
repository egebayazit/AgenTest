#pragma once
#include <string>

namespace web_bridge {

struct WebSnapshotResult {
    bool success = false;
    std::string jsonOutput;
    std::string errorMessage;
};

// Python scriptini çalıştırır ve çıktısını alır
WebSnapshotResult CaptureWebSnapshot();

} // namespace web_bridge