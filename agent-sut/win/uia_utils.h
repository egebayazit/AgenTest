#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

// --- Include order matters for COM/Automation ---
#include <Windows.h>
#include <rpc.h>
#include <rpcndr.h>
#include <ole2.h>
#include <oleacc.h>
#include <Unknwn.h>

#include <UIAutomationCore.h>
#include <UIAutomationClient.h>
#include <UIAutomation.h>

#include <string>
#include <vector>

// UI Automation Pattern Support
struct UiaPatterns {
  bool invoke = false;
  bool value = false;
  bool selectionItem = false;
  bool toggle = false;
  bool expandCollapse = false;
  bool scroll = false;
  bool text = false;
  bool legacy = false;  // IAccessible pattern support
};

// UI Element Information
struct UiaElem {
  // Identity & Classification
  std::wstring name;
  std::wstring automationId; // EKLENDİ: Otomasyon Kimliği
  std::wstring className;
  std::wstring controlType;
  std::wstring value;        // EKLENDİ: Input/Edit alanlarındaki metin değeri
  
  // Position & Visibility
  RECT rect{};              // Bounding rectangle (screen coordinates)
  bool enabled = false;     // IsEnabled property
  bool visible = false;     // Has positive area (calculated)
  bool isOffscreen = false; // IsOffscreen property
  
  // Window & Process Info
  HWND hwnd{};              // Native window handle (if available)
  DWORD pid = 0;            // Process ID
  
  // Interaction Patterns
  UiaPatterns patterns{};
  
  // Hierarchical Path (root → ... → this node)
  std::vector<std::wstring> path;
};

// Main UI Automation Session Manager
class UiaSession {
public:
  UiaSession();
  ~UiaSession();
  
  // Full snapshot (Geriye dönük uyumluluk için, genelde kullanılmaz)
  std::vector<UiaElem> snapshot(int max_elems = 500) const;
  
  // Filtered snapshot optimized for UI automation
  // - Filters offscreen/oversized elements
  // - Focus on foreground window
  // - Deep scans containers (panels, dialogs, tabs)
  std::vector<UiaElem> snapshot_filtered(int max_elems = 500) const;

  // Screen dimensions (primary monitor)
  static int ScreenW();
  static int ScreenH();

private:
  IUIAutomation* uia_ = nullptr;
};

// Helper: Convert UIA ControlType ID to string
std::wstring ControlTypeToString(long ctl);