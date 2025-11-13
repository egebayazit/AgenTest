// uia_utils.cpp - SYSTEMATIC UNIVERSAL VERSION
// Goal: Fast, reliable element discovery for ANY Windows application
// Method: Adaptive tree walking with intelligent pruning (no framework-specific hacks)
#include "uia_utils.h"
#include <OleAuto.h>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>
#include <oleacc.h>
#include <chrono>

// ---- Helpers ----
static std::wstring FromBSTR(BSTR b){ return b? std::wstring(b, SysStringLen(b)) : L""; }
static inline bool HasPositiveArea(const RECT& r){ return (r.right > r.left) && (r.bottom > r.top); }

static inline RECT VirtualScreenRect(){
    RECT r{};
    r.left   = GetSystemMetrics(SM_XVIRTUALSCREEN);
    r.top    = GetSystemMetrics(SM_YVIRTUALSCREEN);
    r.right  = r.left + GetSystemMetrics(SM_CXVIRTUALSCREEN);
    r.bottom = r.top  + GetSystemMetrics(SM_CYVIRTUALSCREEN);
    return r;
}

static inline bool IntersectNonEmpty(const RECT& a, const RECT& b){
    RECT out{};
    if(IntersectRect(&out, &a, &b)) return HasPositiveArea(out);
    return false;
}

static bool IsOwnedBy(HWND candidateRoot, HWND targetRoot){
    if (!candidateRoot || !targetRoot) return false;
    if (candidateRoot == targetRoot) return true;

    HWND owner = GetWindow(candidateRoot, GW_OWNER);
    if (owner){
        HWND ownerRoot = GetAncestor(owner, GA_ROOT);
        if(!ownerRoot) ownerRoot = owner;
        if (ownerRoot == targetRoot) return true;
        if (IsOwnedBy(ownerRoot, targetRoot)) return true;
    }

    DWORD candPid = 0, targetPid = 0;
    GetWindowThreadProcessId(candidateRoot, &candPid);
    GetWindowThreadProcessId(targetRoot, &targetPid);
    if (candPid && candPid == targetPid){
        wchar_t cls[128]{};
        if (GetClassNameW(candidateRoot, cls, 128)){
            if (wcscmp(cls, L"#32768") == 0) return true;
            if (wcsncmp(cls, L"Qt5QWindow", 10) == 0) return true;
            if (wcsncmp(cls, L"Qt6QWindow", 10) == 0) return true;
        }
        LONG_PTR style = GetWindowLongPtr(candidateRoot, GWL_STYLE);
        if (style & WS_POPUP) return true;
    }
    return false;
}

struct UiaWindowCollectorData {
    HWND targetRoot{};
    std::unordered_set<HWND>* allowed{};
};

static BOOL CALLBACK EnumOwnedWindowsProc(HWND hwnd, LPARAM lp){
    auto* data = reinterpret_cast<UiaWindowCollectorData*>(lp);
    if(!data || !data->allowed) return TRUE;

    HWND root = GetAncestor(hwnd, GA_ROOT);
    if(!root) root = hwnd;

    if(data->targetRoot && IsOwnedBy(root, data->targetRoot)){
        data->allowed->insert(root);
    }
    return TRUE;
}

static std::wstring RuntimeIdString(IUIAutomationElement* element){
    SAFEARRAY* runtimeId = nullptr;
    std::wstring key;
    if(SUCCEEDED(element->GetRuntimeId(&runtimeId)) && runtimeId){
        LONG lBound = 0, uBound = -1;
        if(SUCCEEDED(SafeArrayGetLBound(runtimeId, 1, &lBound)) &&
           SUCCEEDED(SafeArrayGetUBound(runtimeId, 1, &uBound))){
            for(LONG i = lBound; i <= uBound; ++i){
                LONG value = 0;
                if(SUCCEEDED(SafeArrayGetElement(runtimeId, &i, &value))){
                    key.append(std::to_wstring(value));
                    key.push_back(L'.');
                }
            }
        }
        SafeArrayDestroy(runtimeId);
    }
    return key;
}

// ---- UiaSession ----
UiaSession::UiaSession() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    CoCreateInstance(CLSID_CUIAutomation, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&uia_));
}
UiaSession::~UiaSession(){
    if(uia_) uia_->Release();
    CoUninitialize();
}
int UiaSession::ScreenW(){ return GetSystemMetrics(SM_CXSCREEN); }
int UiaSession::ScreenH(){ return GetSystemMetrics(SM_CYSCREEN); }

// ---- ControlType string ----
std::wstring ControlTypeToString(long ctl){
    switch(ctl){
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
        default:                          return L"Other";
    }
}

// ---- Patterns ----
static bool HasPattern(IUIAutomationElement* e, PATTERNID pid){
    IUnknown* u = nullptr;
    HRESULT hr = e->GetCurrentPattern(pid, &u);
    if(SUCCEEDED(hr) && u){ u->Release(); return true; }
    return false;
}
static void FillPatterns(IUIAutomationElement* e, UiaPatterns& p){
    p.invoke = HasPattern(e, UIA_InvokePatternId);
    p.value  = HasPattern(e, UIA_ValuePatternId);
    p.selectionItem = HasPattern(e, UIA_SelectionItemPatternId);
    p.toggle = HasPattern(e, UIA_TogglePatternId);
    p.expandCollapse = HasPattern(e, UIA_ExpandCollapsePatternId);
    p.scroll = HasPattern(e, UIA_ScrollPatternId);
    p.text   = HasPattern(e, UIA_TextPatternId);
    p.legacy = HasPattern(e, UIA_LegacyIAccessiblePatternId);
}

// ---- Path generation ----
static std::vector<std::wstring> BuildPath(IUIAutomation* uia, IUIAutomationElement* node){
    std::vector<std::wstring> rev;
    if(!uia || !node) return rev;

    IUIAutomationTreeWalker* walker=nullptr;
    if(FAILED(uia->get_RawViewWalker(&walker)) || !walker) return rev;

    IUIAutomationElement* cur = node;
    for(int depth=0; cur && depth<6; ++depth){
        VARIANT v; VariantInit(&v);
        std::wstring name, cls, ctype;

        cur->GetCurrentPropertyValue(UIA_NamePropertyId, &v);       name  = FromBSTR(v.bstrVal);  VariantClear(&v);
        cur->GetCurrentPropertyValue(UIA_ClassNamePropertyId, &v);  cls   = FromBSTR(v.bstrVal);  VariantClear(&v);
        cur->GetCurrentPropertyValue(UIA_ControlTypePropertyId, &v); ctype = ControlTypeToString(v.lVal); VariantClear(&v);

        std::wstring label;
        if(!ctype.empty()) label += ctype;
        if(!name.empty()){
            if(!label.empty()) label+=L"(";
            label+=name; label+=L")";
        } else if(!cls.empty()){
            if(!label.empty()) label+=L"(";
            label+=cls; label+=L")";
        }
        if(label.empty()) label=L"Node";
        rev.push_back(label);

        IUIAutomationElement* parent=nullptr;
        walker->GetParentElement(cur, &parent);
        if(cur!=node) cur->Release();
        cur = parent;
    }
    if(cur && cur!=node) cur->Release();
    walker->Release();

    std::reverse(rev.begin(), rev.end());
    return rev;
}

// ---- Simple snapshot ----
std::vector<UiaElem> UiaSession::snapshot(int max_elems) const {
    std::vector<UiaElem> out;
    if(!uia_) return out;

    IUIAutomationElement* root=nullptr;
    if (FAILED(uia_->GetRootElement(&root)) || !root) return out;

    IUIAutomationCondition* cond=nullptr; uia_->CreateTrueCondition(&cond);
    IUIAutomationElementArray* arr=nullptr;
    HRESULT hr = root->FindAll(TreeScope_Subtree, cond, &arr);

    if (SUCCEEDED(hr) && arr){
        int len=0; arr->get_Length(&len);
        len = (std::min)(len, max_elems);
        for(int i=0;i<len;i++){
            IUIAutomationElement* e=nullptr; arr->GetElement(i,&e);
            if(!e) continue;

            UiaElem u{};
            VARIANT v; VariantInit(&v);

            e->GetCurrentPropertyValue(UIA_NamePropertyId, &v);             u.name = FromBSTR(v.bstrVal); VariantClear(&v);
            e->GetCurrentPropertyValue(UIA_AutomationIdPropertyId, &v);     u.automationId = FromBSTR(v.bstrVal); VariantClear(&v);
            e->GetCurrentPropertyValue(UIA_ClassNamePropertyId, &v);        u.className = FromBSTR(v.bstrVal); VariantClear(&v);
            e->GetCurrentPropertyValue(UIA_ControlTypePropertyId, &v);      u.controlType = ControlTypeToString(v.lVal); VariantClear(&v);

            e->GetCurrentPropertyValue(UIA_BoundingRectanglePropertyId, &v);
            if ((v.vt & VT_ARRAY) && v.parray){
                SAFEARRAY* sa = v.parray;
                if(sa->cDims==1 && sa->rgsabound[0].cElements==4){
                    double* p=nullptr; SafeArrayAccessData(sa,(void**)&p);
                    u.rect.left=(LONG)p[0];
                    u.rect.top=(LONG)p[1];
                    u.rect.right=(LONG)(p[0] + p[2]);
                    u.rect.bottom=(LONG)(p[1] + p[3]);
                    SafeArrayUnaccessData(sa);
                }
            }
            VariantClear(&v);

            e->GetCurrentPropertyValue(UIA_IsEnabledPropertyId, &v);        u.enabled = (v.vt==VT_BOOL && v.boolVal==VARIANT_TRUE); VariantClear(&v);
            e->GetCurrentPropertyValue(UIA_IsOffscreenPropertyId, &v);      u.isOffscreen = (v.vt==VT_BOOL && v.boolVal==VARIANT_TRUE); VariantClear(&v);
            e->GetCurrentPropertyValue(UIA_ProcessIdPropertyId, &v);        if(v.vt==VT_I4 || v.vt==VT_INT) u.pid=(DWORD)v.lVal; VariantClear(&v);
            e->GetCurrentPropertyValue(UIA_NativeWindowHandlePropertyId, &v); if(v.vt==VT_I4 || v.vt==VT_INT) u.hwnd=(HWND)(INT_PTR)v.lVal; VariantClear(&v);

            u.visible = HasPositiveArea(u.rect);
            FillPatterns(e, u.patterns);
            u.path = BuildPath(uia_, e);

            out.push_back(std::move(u));
            e->Release();
        }
        arr->Release();
    }
    if(cond) cond->Release();
    root->Release();
    return out;
}

// ---- SYSTEMATIC FILTERED SCAN ----
// Core Principles:
// 1. Use TreeWalker (not FindAll) - O(n) instead of O(n²)
// 2. Adaptive pruning based on element characteristics (not framework detection)
// 3. Time-based cutoff for safety
// 4. Smart child limit based on observed patterns
std::vector<UiaElem> UiaSession::snapshot_filtered(int max_elems) const {
    std::vector<UiaElem> out;
    if(!uia_ || max_elems <= 0) return out;

    auto startTime = std::chrono::steady_clock::now();
    const auto MAX_SCAN_TIME = std::chrono::seconds(5);  // Hard timeout

    const RECT vs = VirtualScreenRect();
    const int vsW = vs.right - vs.left;
    const int vsH = vs.bottom - vs.top;

    HWND fgRoot = GetAncestor(GetForegroundWindow(), GA_ROOT);
    if(!fgRoot) fgRoot = GetForegroundWindow();

    std::unordered_set<HWND> allowedRoots;
    if(fgRoot) allowedRoots.insert(fgRoot);

    if(fgRoot){
        UiaWindowCollectorData data{fgRoot, &allowedRoots};
        DWORD threadId = GetWindowThreadProcessId(fgRoot, nullptr);
        if(threadId) EnumThreadWindows(threadId, EnumOwnedWindowsProc, reinterpret_cast<LPARAM>(&data));
        EnumWindows(EnumOwnedWindowsProc, reinterpret_cast<LPARAM>(&data));
    }

    std::vector<HWND> rootHandles;
    rootHandles.reserve(allowedRoots.size() + 1);
    for(HWND h : allowedRoots){
        if(h) rootHandles.push_back(h);
    }

    IUIAutomationElement* globalRoot = nullptr;
    if(rootHandles.empty()){
        if(SUCCEEDED(uia_->GetRootElement(&globalRoot)) && globalRoot){
            rootHandles.push_back(nullptr);
        } else {
            return out;
        }
    }

    std::unordered_set<std::wstring> seenRuntimeIds;

    // UNIVERSAL: No framework detection, just smart heuristics
    struct ElementMetrics {
        LONG controlType;
        std::wstring className;
        std::wstring name;
        bool hasPositiveRect;
        int childCount;  // Estimated
        bool isInteractive;
        bool isContainer;
    };

    auto getMetrics = [](IUIAutomationElement* elem) -> ElementMetrics {
        ElementMetrics m{};
        VARIANT v; VariantInit(&v);
        
        // Control type
        if(SUCCEEDED(elem->GetCurrentPropertyValue(UIA_ControlTypePropertyId, &v))){
            m.controlType = v.lVal;
        }
        VariantClear(&v);
        
        // Class name
        if(SUCCEEDED(elem->GetCurrentPropertyValue(UIA_ClassNamePropertyId, &v))){
            if(v.vt == VT_BSTR && v.bstrVal){
                m.className.assign(v.bstrVal, SysStringLen(v.bstrVal));
            }
        }
        VariantClear(&v);
        
        // Name
        if(SUCCEEDED(elem->GetCurrentPropertyValue(UIA_NamePropertyId, &v))){
            if(v.vt == VT_BSTR && v.bstrVal){
                m.name.assign(v.bstrVal, SysStringLen(v.bstrVal));
            }
        }
        VariantClear(&v);
        
        // Rect
        if(SUCCEEDED(elem->GetCurrentPropertyValue(UIA_BoundingRectanglePropertyId, &v))){
            if((v.vt & VT_ARRAY) && v.parray){
                SAFEARRAY* sa = v.parray;
                if(sa->cDims==1 && sa->rgsabound[0].cElements==4){
                    double* p=nullptr; SafeArrayAccessData(sa,(void**)&p);
                    RECT r;
                    r.left = (LONG)p[0];
                    r.top = (LONG)p[1];
                    r.right = (LONG)(p[0] + p[2]);
                    r.bottom = (LONG)(p[1] + p[3]);
                    m.hasPositiveRect = HasPositiveArea(r);
                    SafeArrayUnaccessData(sa);
                }
            }
        }
        VariantClear(&v);
        
        // Interactive check
        m.isInteractive = (m.controlType == UIA_ButtonControlTypeId ||
                           m.controlType == UIA_EditControlTypeId ||
                           m.controlType == UIA_ComboBoxControlTypeId ||
                           m.controlType == UIA_CheckBoxControlTypeId ||
                           m.controlType == UIA_RadioButtonControlTypeId ||
                           m.controlType == UIA_MenuItemControlTypeId ||
                           m.controlType == UIA_ListItemControlTypeId ||
                           m.controlType == UIA_TabItemControlTypeId ||
                           m.controlType == UIA_HyperlinkControlTypeId);
        
        // Container check
        m.isContainer = (m.controlType == UIA_PaneControlTypeId ||
                         m.controlType == UIA_GroupControlTypeId ||
                         m.controlType == UIA_WindowControlTypeId ||
                         m.controlType == UIA_TabControlTypeId ||
                         m.controlType == UIA_TableControlTypeId ||
                         m.controlType == UIA_ListControlTypeId ||
                         m.controlType == UIA_TreeControlTypeId ||
                         m.className.find(L"Container") != std::wstring::npos ||
                         m.className.find(L"Panel") != std::wstring::npos ||
                         m.className.find(L"View") != std::wstring::npos ||
                         m.className.find(L"Widget") != std::wstring::npos ||
                         m.className.find(L"Frame") != std::wstring::npos ||
                         m.className.find(L"Dialog") != std::wstring::npos);
        
        return m;
    };

    auto shouldPrune = [](const ElementMetrics& m, int depth, int siblingsSoFar) -> bool {
        // Prune if too deep and not important
        if(depth > 12 && !m.isInteractive && !m.isContainer){
            return true;
        }
        
        // Prune if too many siblings (likely a generated list/grid)
        if(siblingsSoFar > 200 && !m.isInteractive){
            return true;
        }
        
        // Prune decorative elements
        if(m.controlType == UIA_ImageControlTypeId && m.name.empty()){
            return true;
        }
        
        // Prune invisible text blocks
        if(m.controlType == UIA_TextControlTypeId && !m.hasPositiveRect){
            return true;
        }
        
        return false;
    };

    auto extractElement = [&](IUIAutomationElement* element, UiaElem& u) -> bool {
        VARIANT v; VariantInit(&v);

        // Rect
        element->GetCurrentPropertyValue(UIA_BoundingRectanglePropertyId, &v);
        if ((v.vt & VT_ARRAY) && v.parray){
            SAFEARRAY* sa = v.parray;
            if(sa->cDims==1 && sa->rgsabound[0].cElements==4){
                double* p=nullptr; SafeArrayAccessData(sa,(void**)&p);
                u.rect.left=(LONG)p[0];
                u.rect.top=(LONG)p[1];
                u.rect.right=(LONG)(p[0] + p[2]);
                u.rect.bottom=(LONG)(p[1] + p[3]);
                SafeArrayUnaccessData(sa);
            }
        }
        VariantClear(&v);

        const bool posArea = HasPositiveArea(u.rect);
        const bool onScreen = posArea && IntersectNonEmpty(u.rect, vs);
        const int  w = u.rect.right - u.rect.left;
        const int  h = u.rect.bottom - u.rect.top;

        // Filter offscreen/oversized
        if(!onScreen && posArea){  // Has rect but offscreen
            return false;
        }
        if(w > (int)(vsW*2) || h > (int)(vsH*2)){  // Unreasonably large
            return false;
        }

        // Properties
        element->GetCurrentPropertyValue(UIA_NamePropertyId, &v);
        if(v.vt==VT_BSTR && v.bstrVal) u.name = FromBSTR(v.bstrVal);
        VariantClear(&v);

        element->GetCurrentPropertyValue(UIA_ControlTypePropertyId, &v);
        u.controlType = ControlTypeToString(v.lVal);
        VariantClear(&v);

        element->GetCurrentPropertyValue(UIA_ClassNamePropertyId, &v);
        if(v.vt==VT_BSTR && v.bstrVal) u.className = FromBSTR(v.bstrVal);
        VariantClear(&v);

        element->GetCurrentPropertyValue(UIA_AutomationIdPropertyId, &v);
        if(v.vt==VT_BSTR && v.bstrVal) u.automationId = FromBSTR(v.bstrVal);
        VariantClear(&v);

        element->GetCurrentPropertyValue(UIA_IsEnabledPropertyId, &v);
        u.enabled = (v.vt==VT_BOOL && v.boolVal==VARIANT_TRUE);
        VariantClear(&v);

        element->GetCurrentPropertyValue(UIA_NativeWindowHandlePropertyId, &v);
        if(v.vt==VT_I4 || v.vt==VT_INT) u.hwnd=(HWND)(INT_PTR)v.lVal;
        VariantClear(&v);

        u.visible = posArea;
        FillPatterns(element, u.patterns);
        u.path = BuildPath(uia_, element);
        return true;
    };
    
    IUIAutomationTreeWalker* walker = nullptr;
    uia_->get_RawViewWalker(&walker);
    if(!walker) return out;

    // UNIVERSAL BREADTH-FIRST WALKER with adaptive pruning
    for(HWND rootHandle : rootHandles){
        if((int)out.size() >= max_elems) break;
        
        auto now = std::chrono::steady_clock::now();
        if(now - startTime > MAX_SCAN_TIME) break;  // Timeout
        
        IUIAutomationElement* base = nullptr;
        if(rootHandle){
            if(FAILED(uia_->ElementFromHandle(rootHandle, &base)) || !base){
                continue;
            }
        } else {
            base = globalRoot;
            if(!base) continue;
            base->AddRef();
        }

        // BFS with pruning
        std::vector<std::tuple<IUIAutomationElement*, int, int>> queue;  // {element, depth, siblingIndex}
        queue.push_back({base, 0, 0});

        while(!queue.empty() && (int)out.size() < max_elems){
            auto now = std::chrono::steady_clock::now();
            if(now - startTime > MAX_SCAN_TIME){
                // Cleanup and exit
                for(auto& [elem, _, __] : queue){
                    elem->Release();
                }
                break;
            }
            
            auto [elem, depth, siblingIdx] = queue.front();
            queue.erase(queue.begin());
            
            std::wstring runtimeKey = RuntimeIdString(elem);
            bool firstTime = runtimeKey.empty() || seenRuntimeIds.insert(runtimeKey).second;
            
            if(firstTime){
                // Get metrics for pruning decision
                ElementMetrics metrics = getMetrics(elem);
                
                if(!shouldPrune(metrics, depth, siblingIdx)){
                    UiaElem u{};
                    if(extractElement(elem, u)){
                        out.push_back(std::move(u));
                        
                        // Explore children if container or interactive
                        if(metrics.isContainer || metrics.isInteractive || depth < 3){
                            IUIAutomationElement* child = nullptr;
                            if(SUCCEEDED(walker->GetFirstChildElement(elem, &child)) && child){
                                int childSiblingIdx = 0;
                                IUIAutomationElement* current = child;
                                
                                while(current && childSiblingIdx < 300){  // Max 300 siblings
                                    queue.push_back({current, depth + 1, childSiblingIdx});
                                    childSiblingIdx++;
                                    
                                    IUIAutomationElement* next = nullptr;
                                    if(SUCCEEDED(walker->GetNextSiblingElement(current, &next)) && next){
                                        current = next;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            elem->Release();
        }
    }

    if(walker) walker->Release();
    if(globalRoot) globalRoot->Release();
    return out;
}