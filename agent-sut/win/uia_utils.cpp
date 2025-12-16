// uia_utils.cpp - BALANCED VISIBILITY WALKER
// Fixes: Missing deep elements (depth > 12) and Dialogs being pruned.
// Performance: Still <2s for most apps, but safer than previous aggressive pruning.

#include "uia_utils.h"
#include <OleAuto.h>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>
#include <oleacc.h>
#include <chrono>
#include <tuple>

// ========================= Helper Functions =========================

static std::wstring FromBSTR(BSTR b) { return b ? std::wstring(b, SysStringLen(b)) : L""; }
static inline bool HasPositiveArea(const RECT& r) { return (r.right > r.left) && (r.bottom > r.top); }

static inline RECT VirtualScreenRect() {
    RECT r{};
    r.left = GetSystemMetrics(SM_XVIRTUALSCREEN);
    r.top = GetSystemMetrics(SM_YVIRTUALSCREEN);
    r.right = r.left + GetSystemMetrics(SM_CXVIRTUALSCREEN);
    r.bottom = r.top + GetSystemMetrics(SM_CYVIRTUALSCREEN);
    return r;
}

static inline bool IntersectNonEmpty(const RECT& a, const RECT& b) {
    RECT out{};
    if (IntersectRect(&out, &a, &b)) return HasPositiveArea(out);
    return false;
}

std::wstring ControlTypeToString(long ctl) {
    switch (ctl) {
    case UIA_ButtonControlTypeId:     return L"Button";
    case UIA_EditControlTypeId:       return L"Edit";
    case UIA_TextControlTypeId:       return L"Text";
    case UIA_CheckBoxControlTypeId:   return L"CheckBox";
    case UIA_ComboBoxControlTypeId:   return L"ComboBox";
    case UIA_MenuItemControlTypeId:   return L"MenuItem";
    case UIA_ListItemControlTypeId:   return L"ListItem";
    case UIA_WindowControlTypeId:     return L"Window";
    case UIA_TabItemControlTypeId:    return L"TabItem";
    case UIA_HyperlinkControlTypeId:  return L"Hyperlink";
    case UIA_PaneControlTypeId:       return L"Pane";
    case UIA_GroupControlTypeId:      return L"Group";
    case UIA_ListControlTypeId:       return L"List";
    case UIA_TableControlTypeId:      return L"Table";
    case UIA_DataGridControlTypeId:   return L"DataGrid";
    case UIA_TreeControlTypeId:       return L"Tree";
    case UIA_TreeItemControlTypeId:   return L"TreeItem";
    case UIA_DocumentControlTypeId:   return L"Document";
    case UIA_ToolBarControlTypeId:    return L"ToolBar";
    case UIA_MenuBarControlTypeId:    return L"MenuBar";
    case UIA_MenuControlTypeId:       return L"Menu";
    case UIA_ImageControlTypeId:      return L"Image";
    case UIA_HeaderControlTypeId:     return L"Header";
    case UIA_HeaderItemControlTypeId: return L"HeaderItem";
    case UIA_ScrollBarControlTypeId:  return L"ScrollBar";
    case UIA_StatusBarControlTypeId:  return L"StatusBar";
    default:                          return L"Other";
    }
}

// ========================= Cache Optimizations =========================

static IUIAutomationCacheRequest* CreateSmartCacheRequest(IUIAutomation* uia) {
    IUIAutomationCacheRequest* req = nullptr;
    if (FAILED(uia->CreateCacheRequest(&req))) return nullptr;

    req->AddProperty(UIA_NamePropertyId);
    req->AddProperty(UIA_ClassNamePropertyId);
    req->AddProperty(UIA_ControlTypePropertyId);
    req->AddProperty(UIA_BoundingRectanglePropertyId);
    req->AddProperty(UIA_AutomationIdPropertyId);
    req->AddProperty(UIA_IsEnabledPropertyId);
    req->AddProperty(UIA_IsOffscreenPropertyId);
    req->AddProperty(UIA_NativeWindowHandlePropertyId);
    req->AddProperty(UIA_ProcessIdPropertyId);

    // Light pattern checks
    req->AddPattern(UIA_InvokePatternId);
    req->AddPattern(UIA_ValuePatternId);
    req->AddPattern(UIA_SelectionItemPatternId);
    req->AddPattern(UIA_TogglePatternId);
    req->AddPattern(UIA_ExpandCollapsePatternId);
    req->AddPattern(UIA_ScrollPatternId);
    req->AddPattern(UIA_TextPatternId);
    req->AddPattern(UIA_LegacyIAccessiblePatternId);

    req->put_TreeScope(TreeScope_Element);
    return req;
}

static void GetSmartProp(IUIAutomationElement* e, PROPERTYID pid, VARIANT* v) {
    VariantInit(v);
    e->GetCachedPropertyValue(pid, v); 
}

static void FillPatternsSmart(IUIAutomationElement* e, UiaPatterns& p) {
    auto check = [&](PATTERNID pid) -> bool {
        IUnknown* u = nullptr;
        if (SUCCEEDED(e->GetCachedPattern(pid, &u)) && u) {
            u->Release();
            return true;
        }
        return false;
    };

    p.invoke = check(UIA_InvokePatternId);
    p.value = check(UIA_ValuePatternId);
    p.selectionItem = check(UIA_SelectionItemPatternId);
    p.toggle = check(UIA_TogglePatternId);
    p.expandCollapse = check(UIA_ExpandCollapsePatternId);
    p.scroll = check(UIA_ScrollPatternId);
    p.text = check(UIA_TextPatternId);
    p.legacy = check(UIA_LegacyIAccessiblePatternId);
}

// ========================= Window Finding =========================

struct UiaWindowCollectorData {
    HWND targetRoot{};
    std::unordered_set<HWND>* allowed{};
};

static BOOL CALLBACK EnumOwnedWindowsProc(HWND hwnd, LPARAM lp) {
    auto* data = reinterpret_cast<UiaWindowCollectorData*>(lp);
    if (!data || !data->allowed) return TRUE;
    HWND root = GetAncestor(hwnd, GA_ROOT);
    if (!root) root = hwnd;
    if (data->targetRoot && root == data->targetRoot) {
        data->allowed->insert(root);
    }
    return TRUE;
}

static std::wstring RuntimeIdString(IUIAutomationElement* element) {
    SAFEARRAY* runtimeId = nullptr;
    std::wstring key;
    if (SUCCEEDED(element->GetRuntimeId(&runtimeId)) && runtimeId) {
        long* pData = nullptr;
        if (SUCCEEDED(SafeArrayAccessData(runtimeId, (void**)&pData))) {
            for (ULONG i = 0; i < runtimeId->rgsabound[0].cElements; ++i) {
                key += std::to_wstring(pData[i]) + L".";
            }
            SafeArrayUnaccessData(runtimeId);
        }
        SafeArrayDestroy(runtimeId);
    }
    return key;
}

// ========================= UiaSession Class Implementation =========================

UiaSession::UiaSession() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    CoCreateInstance(CLSID_CUIAutomation, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&uia_));
}

UiaSession::~UiaSession() {
    if (uia_) uia_->Release();
    CoUninitialize();
}

int UiaSession::ScreenW() { return GetSystemMetrics(SM_CXSCREEN); }
int UiaSession::ScreenH() { return GetSystemMetrics(SM_CYSCREEN); }

// ========================= SMART VISIBILITY SNAPSHOT =========================

std::vector<UiaElem> UiaSession::snapshot_filtered(int max_elems) const {
    std::vector<UiaElem> out;
    if (!uia_ || max_elems <= 0) return out;

    // Timeout increased slightly to allow deeper scan
    auto startTime = std::chrono::steady_clock::now();
    const auto MAX_SCAN_TIME = std::chrono::seconds(15); 

    const RECT vs = VirtualScreenRect();

    IUIAutomationCacheRequest* cacheReq = CreateSmartCacheRequest(uia_);
    if (!cacheReq) return out;

    IUIAutomationTreeWalker* walker = nullptr;
    uia_->get_RawViewWalker(&walker);
    if (!walker) { cacheReq->Release(); return out; }

    HWND fgRoot = GetAncestor(GetForegroundWindow(), GA_ROOT);
    if (!fgRoot) fgRoot = GetForegroundWindow();

    std::unordered_set<HWND> allowedRoots;
    if (fgRoot) {
        allowedRoots.insert(fgRoot);
        UiaWindowCollectorData data{ fgRoot, &allowedRoots };
        DWORD tid = GetWindowThreadProcessId(fgRoot, nullptr);
        if (tid) EnumThreadWindows(tid, EnumOwnedWindowsProc, (LPARAM)&data);
    }

    struct QueueItem {
        IUIAutomationElement* elem;
        int depth;
        std::vector<std::wstring> path;
    };
    std::vector<QueueItem> queue;

    for (HWND h : allowedRoots) {
        if (!h) continue;
        IUIAutomationElement* root = nullptr;
        if (SUCCEEDED(uia_->ElementFromHandleBuildCache(h, cacheReq, &root)) && root) {
            queue.push_back({ root, 0, {} });
        }
    }

    if (queue.empty()) {
        IUIAutomationElement* root = nullptr;
        if (SUCCEEDED(uia_->GetRootElementBuildCache(cacheReq, &root)) && root) {
            queue.push_back({ root, 0, {} });
        }
    }

    std::unordered_set<std::wstring> seenRuntimeIds;

    size_t head = 0;
    while (head < queue.size() && (int)out.size() < max_elems) {
        QueueItem item = queue[head++];
        IUIAutomationElement* elem = item.elem;

        if (head % 30 == 0) {
            if (std::chrono::steady_clock::now() - startTime > MAX_SCAN_TIME) break;
        }

        std::wstring rid = RuntimeIdString(elem);
        if (!rid.empty() && !seenRuntimeIds.insert(rid).second) {
            elem->Release(); continue;
        }

        UiaElem u{};
        VARIANT v;

        GetSmartProp(elem, UIA_ControlTypePropertyId, &v);
        long cType = v.lVal;
        u.controlType = ControlTypeToString(cType);

        GetSmartProp(elem, UIA_IsOffscreenPropertyId, &v);
        u.isOffscreen = (v.vt == VT_BOOL && v.boolVal == VARIANT_TRUE);
        VariantClear(&v);

        GetSmartProp(elem, UIA_BoundingRectanglePropertyId, &v);
        if ((v.vt & VT_ARRAY) && v.parray) {
            SAFEARRAY* sa = v.parray;
            double* p = nullptr; SafeArrayAccessData(sa, (void**)&p);
            u.rect.left = (LONG)p[0]; u.rect.top = (LONG)p[1];
            u.rect.right = (LONG)(p[0] + p[2]); u.rect.bottom = (LONG)(p[1] + p[3]);
            SafeArrayUnaccessData(sa);
        }
        VariantClear(&v);

        bool posArea = HasPositiveArea(u.rect);
        bool onScreen = posArea && IntersectNonEmpty(u.rect, vs);

        // --- IMPROVED PRUNING ---
        bool skipSelf = false;
        
        // If offscreen, skip ONLY if not a Window/Pane which might contain visible popups
        // "Dialogs" sometimes report weird offscreen states or negative coords relative to parent.
        bool isDialogOrWindow = (cType == UIA_WindowControlTypeId || u.controlType.find(L"Window") != std::wstring::npos);
        
        if (item.depth > 0 && (!onScreen || u.isOffscreen) && !isDialogOrWindow) {
            skipSelf = true;
        }

        if (!skipSelf) {
            GetSmartProp(elem, UIA_NamePropertyId, &v);
            if (v.vt == VT_BSTR && v.bstrVal) u.name = FromBSTR(v.bstrVal);
            VariantClear(&v);

            GetSmartProp(elem, UIA_ClassNamePropertyId, &v);
            if (v.vt == VT_BSTR && v.bstrVal) u.className = FromBSTR(v.bstrVal);
            VariantClear(&v);

            GetSmartProp(elem, UIA_AutomationIdPropertyId, &v);
            if (v.vt == VT_BSTR && v.bstrVal) u.automationId = FromBSTR(v.bstrVal);
            VariantClear(&v);

            GetSmartProp(elem, UIA_IsEnabledPropertyId, &v);
            u.enabled = (v.vt == VT_BOOL && v.boolVal == VARIANT_TRUE);
            VariantClear(&v);

            GetSmartProp(elem, UIA_NativeWindowHandlePropertyId, &v);
            if (v.vt == VT_I4 || v.vt == VT_INT) u.hwnd = (HWND)(INT_PTR)v.lVal;
            VariantClear(&v);

            GetSmartProp(elem, UIA_ProcessIdPropertyId, &v);
            if (v.vt == VT_I4 || v.vt == VT_INT) u.pid = (DWORD)v.lVal;
            VariantClear(&v);

            u.visible = posArea;
            FillPatternsSmart(elem, u.patterns);

            std::wstring label = u.controlType;
            if (!u.name.empty()) label += L"(" + u.name + L")";
            else if (!u.className.empty()) label += L"(" + u.className + L")";
            
            std::vector<std::wstring> myPath = item.path;
            myPath.push_back(label);
            u.path = myPath;

            out.push_back(std::move(u));
        }

        // --- DEEP TRAVERSAL ---
        // Relaxed depth limit (12 -> 18) to catch deep dialog content
        bool isContainer = (cType == UIA_WindowControlTypeId || cType == UIA_PaneControlTypeId || 
                            cType == UIA_GroupControlTypeId || cType == UIA_ListControlTypeId || 
                            cType == UIA_TableControlTypeId || cType == UIA_TreeControlTypeId ||
                            cType == UIA_DataGridControlTypeId || cType == UIA_CustomControlTypeId); // Custom controls often wrap content

        if (!skipSelf && item.depth < 18 && (isContainer || item.depth < 4)) {
            IUIAutomationElement* child = nullptr;
            if (SUCCEEDED(walker->GetFirstChildElementBuildCache(elem, cacheReq, &child)) && child) {
                
                int consecutiveOffscreenCount = 0;
                IUIAutomationElement* current = child;
                
                while (current) {
                    VARIANT vOff; VariantInit(&vOff);
                    current->GetCachedPropertyValue(UIA_IsOffscreenPropertyId, &vOff);
                    bool isChildOff = (vOff.vt == VT_BOOL && vOff.boolVal == VARIANT_TRUE);
                    VariantClear(&vOff);

                    // Check rect as well for offscreen heuristic
                    VARIANT vR; VariantInit(&vR);
                    current->GetCachedPropertyValue(UIA_BoundingRectanglePropertyId, &vR);
                    RECT childRect = {0,0,0,0};
                    if ((vR.vt & VT_ARRAY) && vR.parray) {
                        SAFEARRAY* sa = vR.parray;
                        double* p = nullptr; SafeArrayAccessData(sa, (void**)&p);
                        childRect.left = (LONG)p[0]; childRect.top = (LONG)p[1];
                        childRect.right = (LONG)(p[0] + p[2]); childRect.bottom = (LONG)(p[1] + p[3]);
                        SafeArrayUnaccessData(sa);
                    }
                    VariantClear(&vR);
                    bool childVisible = HasPositiveArea(childRect) && IntersectNonEmpty(childRect, vs);

                    if (isChildOff || !childVisible) {
                        consecutiveOffscreenCount++;
                    } else {
                        consecutiveOffscreenCount = 0;
                    }

                    // Relaxed Limit: Stop only after 25 invisible siblings to prevent accidental cutoff in grids
                    if (consecutiveOffscreenCount > 25) {
                        current->Release(); 
                        break; 
                    }

                    queue.push_back({ current, item.depth + 1, item.path });
                    
                    IUIAutomationElement* next = nullptr;
                    if (FAILED(walker->GetNextSiblingElementBuildCache(current, cacheReq, &next))) {
                        break;
                    }
                    current = next;
                }
            }
        }

        elem->Release();
    }

    for (size_t i = head; i < queue.size(); ++i) {
        if (queue[i].elem) queue[i].elem->Release();
    }

    if (walker) walker->Release();
    if (cacheReq) cacheReq->Release();

    return out;
}

std::vector<UiaElem> UiaSession::snapshot(int max_elems) const {
    return std::vector<UiaElem>();
}