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
#include <UIAutomation.h>

#include <string>
#include <vector>

struct UiaPatterns {
  bool invoke=false, value=false, selectionItem=false, toggle=false,
       expandCollapse=false, scroll=false, text=false, legacy=false;
};

struct UiaElem {
  std::wstring name, automationId, className, controlType;
  RECT  rect{};         // UIA bounding rect
  bool  enabled=false;
  bool  visible=false;  // positive area
  bool  isOffscreen=false;
  HWND  hwnd{};
  DWORD pid=0;
  UiaPatterns patterns{};
  std::vector<std::wstring> path; // kısa hiyerarşik yol (root->...->node)
};

class UiaSession {
public:
  UiaSession();
  ~UiaSession();
  std::vector<UiaElem> snapshot(int max_elems = 300) const;
  std::vector<UiaElem> snapshot_filtered(int max_elems = 120) const;

  // ekran (birincil) boyutu
  static int ScreenW();
  static int ScreenH();

private:
  IUIAutomation* uia_ = nullptr;
};

std::wstring ControlTypeToString(long ctl);
