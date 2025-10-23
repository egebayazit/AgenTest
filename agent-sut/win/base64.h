// ...existing code...
#pragma once
#include <string>

std::string base64_encode(const unsigned char* bytes, size_t len);
std::string base64_decode(const std::string& encoded);
// ...existing code...
