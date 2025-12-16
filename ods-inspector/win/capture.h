#pragma once
#define WIN32_LEAN_AND_MEAN
#include <vector>
#include <Windows.h>

// Ekranın (birincil monitör) tamamını PNG (BGRA->PNG) olarak hafızaya yazar.
bool CaptureScreenPng(std::vector<unsigned char>& out_png);

// Screenshot boyutlarını döndürür (width, height)
bool GetScreenDimensions(int& width, int& height);