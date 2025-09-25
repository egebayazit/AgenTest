// uia_utils.cpp
#include "uia_utils.h"
#include <OleAuto.h>
#include <algorithm>

// ---- yardımcılar ----
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

// ---- Kısa yol üretimi (root→...→node, max 6 seviye) ----
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

// ---- Tüm öğeler (kullanmıyoruz ama dursun) ----
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
                    u.rect.left=(LONG)p[0]; u.rect.top=(LONG)p[1];
                    u.rect.right=(LONG)p[2]; u.rect.bottom=(LONG)p[3];
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

// ---- Filtreli (lite) ----
std::vector<UiaElem> UiaSession::snapshot_filtered(int max_elems) const {
    std::vector<UiaElem> out;
    if(!uia_) return out;

    IUIAutomationElement* root=nullptr;
    if (FAILED(uia_->GetRootElement(&root)) || !root) return out;

    IUIAutomationCondition* cond=nullptr; uia_->CreateTrueCondition(&cond);
    IUIAutomationElementArray* arr=nullptr;
    HRESULT hr = root->FindAll(TreeScope_Subtree, cond, &arr);

    const RECT vs = VirtualScreenRect();
    const int vsW = vs.right - vs.left;
    const int vsH = vs.bottom - vs.top;
    const HWND fgRoot = GetAncestor(GetForegroundWindow(), GA_ROOT);

    if (SUCCEEDED(hr) && arr){
        int len=0; arr->get_Length(&len);
        for(int i=0;i<len && (int)out.size()<max_elems;i++){
            IUIAutomationElement* e=nullptr; arr->GetElement(i,&e);
            if(!e) continue;

            UiaElem u{};
            VARIANT v; VariantInit(&v);

            // önce rect + ekranda mı?
            e->GetCurrentPropertyValue(UIA_BoundingRectanglePropertyId, &v);
            if ((v.vt & VT_ARRAY) && v.parray){
                SAFEARRAY* sa = v.parray;
                if(sa->cDims==1 && sa->rgsabound[0].cElements==4){
                    double* p=nullptr; SafeArrayAccessData(sa,(void**)&p);
                    u.rect.left=(LONG)p[0]; u.rect.top=(LONG)p[1];
                    u.rect.right=(LONG)p[2]; u.rect.bottom=(LONG)p[3];
                    SafeArrayUnaccessData(sa);
                }
            }
            VariantClear(&v);

            const bool posArea = HasPositiveArea(u.rect);
            const bool onScreen = posArea && IntersectNonEmpty(u.rect, vs);
            const int  w = u.rect.right - u.rect.left;
            const int  h = u.rect.bottom - u.rect.top;

            // 1) ekrana değmiyorsa alma
            // 2) absürt büyükse (ekranın 1.5 katından büyük) alma
            if(!onScreen || w > (int)(vsW*1.5) || h > (int)(vsH*1.5)){
                e->Release(); continue;
            }

            // hwnd & kök
            e->GetCurrentPropertyValue(UIA_NativeWindowHandlePropertyId, &v);
            if(v.vt==VT_I4 || v.vt==VT_INT) u.hwnd=(HWND)(INT_PTR)v.lVal;
            VariantClear(&v);

            if(u.hwnd){
                HWND r = GetAncestor(u.hwnd, GA_ROOT);
                // sadece aktif kök penceredeki öğeler
                if(r != fgRoot){
                    e->Release(); continue;
                }
            }

            // minimal alanlar
            e->GetCurrentPropertyValue(UIA_NamePropertyId, &v);       u.name = FromBSTR(v.bstrVal); VariantClear(&v);
            e->GetCurrentPropertyValue(UIA_ControlTypePropertyId, &v); u.controlType = ControlTypeToString(v.lVal); VariantClear(&v);
            e->GetCurrentPropertyValue(UIA_IsEnabledPropertyId, &v);   u.enabled = (v.vt==VT_BOOL && v.boolVal==VARIANT_TRUE); VariantClear(&v);

            u.visible = true;
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
