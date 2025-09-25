#pragma once
#include <vector>
#include <Windows.h>

// Ekranın (birincil monitör) tamamını PNG (BGRA->PNG) olarak hafızaya yazar.
bool CaptureScreenPng(std::vector<unsigned char>& out_png);
