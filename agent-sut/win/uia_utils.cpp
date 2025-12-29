#include "uia_utils.h"
#include <OleAuto.h>
#include <algorithm>
#include <string>
#include <vector>
#include <oleacc.h>
#include <unordered_set>
#include <iostream>

// ========================= Helper Functions =========================

static std::wstring FromBSTR(BSTR b) { return b ? std::wstring(b, SysStringLen(b)) : L""; }

std::wstring ControlTypeToString(long ctl) {
    switch (ctl) {
    case UIA_ButtonControlTypeId:     return L"Button";
    case UIA_EditControlTypeId:       return L"Edit";
    case UIA_TextControlTypeId:       return L"Text";
    case UIA_CheckBoxControlTypeId:   return L"CheckBox";
    case UIA_RadioButtonControlTypeId:return L"RadioButton";
    case UIA_ComboBoxControlTypeId:   return L"ComboBox";
    case UIA_ListControlTypeId:       return L"List";
    case UIA_ListItemControlTypeId:   return L"ListItem";
    case UIA_WindowControlTypeId:     return L"Window";
    case UIA_PaneControlTypeId:       return L"Pane";
    case UIA_HeaderControlTypeId:     return L"Header";
    case UIA_HeaderItemControlTypeId: return L"HeaderItem";
    case UIA_TableControlTypeId:      return L"Table";
    case UIA_TreeControlTypeId:       return L"Tree";
    case UIA_TreeItemControlTypeId:   return L"TreeItem";
    case UIA_MenuControlTypeId:       return L"Menu";
    case UIA_MenuItemControlTypeId:   return L"MenuItem";
    case UIA_TabControlTypeId:        return L"Tab";
    case UIA_TabItemControlTypeId:    return L"TabItem";
    case UIA_HyperlinkControlTypeId:  return L"Hyperlink";
    case UIA_ImageControlTypeId:      return L"Image";
    case UIA_DocumentControlTypeId:   return L"Document";
    case UIA_GroupControlTypeId:      return L"Group";
    case UIA_CustomControlTypeId:     return L"Custom";
    case UIA_DataGridControlTypeId:   return L"DataGrid";
    case UIA_ToolBarControlTypeId:    return L"ToolBar";
    case UIA_StatusBarControlTypeId:  return L"StatusBar";
    case UIA_ScrollBarControlTypeId:  return L"ScrollBar";
    default:                          return L"Other";
    }
}

static inline bool HasPositiveArea(const RECT& r) { 
    return (r.right > r.left) && (r.bottom > r.top); 
}

// ========================= Element Data Collector =========================

static void CollectElementData(IUIAutomationElement* elem, UiaElem& u) {
    VARIANT v;
    VariantInit(&v);

    // 1. Control Type
    if (SUCCEEDED(elem->GetCachedPropertyValue(UIA_ControlTypePropertyId, &v))) {
        u.controlType = ControlTypeToString(v.lVal);
        VariantClear(&v);
    }

    // 2. Rect
    if (SUCCEEDED(elem->GetCachedPropertyValue(UIA_BoundingRectanglePropertyId, &v))) {
        if ((v.vt & VT_ARRAY) && v.parray) {
            double* p = nullptr;
            SafeArrayAccessData(v.parray, (void**)&p);
            u.rect.left = (LONG)p[0]; 
            u.rect.top = (LONG)p[1];
            u.rect.right = (LONG)(p[0] + p[2]); 
            u.rect.bottom = (LONG)(p[1] + p[3]);
            SafeArrayUnaccessData(v.parray);
        }
        VariantClear(&v);
    }

    // 3. Name
    if (SUCCEEDED(elem->GetCachedPropertyValue(UIA_NamePropertyId, &v)) && v.bstrVal) {
        u.name = FromBSTR(v.bstrVal);
        VariantClear(&v);
    }

    // 4. Automation ID
    if (SUCCEEDED(elem->GetCachedPropertyValue(UIA_AutomationIdPropertyId, &v)) && v.bstrVal) {
        u.automationId = FromBSTR(v.bstrVal);
        VariantClear(&v);
    }

    // 5. Class Name
    if (SUCCEEDED(elem->GetCachedPropertyValue(UIA_ClassNamePropertyId, &v)) && v.bstrVal) {
        u.className = FromBSTR(v.bstrVal);
        VariantClear(&v);
    }

    // 6. Value (Metin) - Senin istediğin özellik
    // CacheRequest'e UIA_ValueValuePropertyId ekledik, direkt oradan çekiyoruz.
    if (SUCCEEDED(elem->GetCachedPropertyValue(UIA_ValueValuePropertyId, &v)) && v.bstrVal) {
        u.value = FromBSTR(v.bstrVal);
        u.patterns.value = true;
        VariantClear(&v);
    }

    // 7. Offscreen & Enabled
    if (SUCCEEDED(elem->GetCachedPropertyValue(UIA_IsOffscreenPropertyId, &v))) {
        u.isOffscreen = (v.boolVal == VARIANT_TRUE);
        VariantClear(&v);
    }
    if (SUCCEEDED(elem->GetCachedPropertyValue(UIA_IsEnabledPropertyId, &v))) {
        u.enabled = (v.boolVal == VARIANT_TRUE);
        VariantClear(&v);
    }

    // 8. Patterns (True/False Flags)
    auto CheckPat = [&](PATTERNID pid) {
        IUnknown* unk = nullptr;
        if (SUCCEEDED(elem->GetCachedPattern(pid, &unk)) && unk) {
            unk->Release(); return true;
        }
        return false;
    };
    u.patterns.invoke = CheckPat(UIA_InvokePatternId);
    u.patterns.toggle = CheckPat(UIA_TogglePatternId);
    u.patterns.selectionItem = CheckPat(UIA_SelectionItemPatternId);
    u.patterns.scroll = CheckPat(UIA_ScrollPatternId);
}

// ========================= Recursive Walker =========================

static void RecursiveWalk(
    IUIAutomationTreeWalker* walker, 
    IUIAutomationElement* parent, 
    IUIAutomationCacheRequest* cacheReq,
    std::vector<UiaElem>& results, 
    int depth, 
    int maxElems) 
{
    if ((int)results.size() >= maxElems || depth > 60) return;

    // Ekran boyutlarını al (Sınır kontrolü için)
    int sw = GetSystemMetrics(SM_CXSCREEN);
    int sh = GetSystemMetrics(SM_CYSCREEN);

    IUIAutomationElement* child = nullptr;
    walker->GetFirstChildElementBuildCache(parent, cacheReq, &child);

    while (child && (int)results.size() < maxElems) {
        UiaElem u{};
        CollectElementData(child, u);

        // --- GÖRÜNÜRLÜK FİLTRESİ ---
        
        // 1. Boyut kontrolü
        bool hasArea = (u.rect.right > u.rect.left) && (u.rect.bottom > u.rect.top);
        
        // 2. Ekran sınırları kontrolü (Scroll yapmadan görülebilir mi?)
        // Elementin merkezi veya bir kısmı ekran sınırları içinde mi?
        bool isInView = (u.rect.left < sw && u.rect.right > 0 && 
                         u.rect.top < sh && u.rect.bottom > 0);

        // 3. UIA Offscreen raporu (isInView ile birlikte kullanıldığında daha güvenlidir)
        // Not: Bazı elementler isInView olsa bile isOffscreen olabilir (başka bir tabda olması gibi)
        bool isTrulyVisible = hasArea && isInView && !u.isOffscreen;

        // Elementi sadece gerçekten ekrandaysa listeye ekle
        if (isTrulyVisible && (!u.name.empty() || !u.automationId.empty() || !u.value.empty())) {
            u.visible = true;
            results.push_back(u);
        }

        // --- İÇERİ DALMA MANTIĞI ---
        // Element ekran dışında olsa bile içindeki çocuklar ekranda olabilir 
        // (Örn: Bir konteynerın sadece alt kısmı ekrandadır). 
        // Bu yüzden kapsayıcıların (Leaf olmayanların) içine girmeye devam ediyoruz.
        bool isLeaf = (u.controlType == L"Image" || u.controlType == L"Text" || u.controlType == L"ScrollBar");
        
        if (!isLeaf) {
            RecursiveWalk(walker, child, cacheReq, results, depth + 1, maxElems);
        }

        IUIAutomationElement* next = nullptr;
        walker->GetNextSiblingElementBuildCache(child, cacheReq, &next);
        child->Release();
        child = next;
    }
}

// ========================= UiaSession Implementation =========================

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

std::vector<UiaElem> UiaSession::snapshot_filtered(int max_elems) const {
    std::vector<UiaElem> out;
    if (!uia_) return out;

    // 1. AKTİF PENCEREYİ BUL (Mouse değil, Foreground Window)
    HWND fgWin = GetForegroundWindow();
    if (!fgWin) return out;

    IUIAutomationElement* root = nullptr;
    if (FAILED(uia_->ElementFromHandle(fgWin, &root)) || !root) return out;

    // 2. Cache Request Hazırla (Bu kısım çok önemli, tüm propertyleri tek seferde ister)
    IUIAutomationCacheRequest* cacheReq = nullptr;
    uia_->CreateCacheRequest(&cacheReq);
    if (cacheReq) {
        // İstenen Özellikler
        cacheReq->AddProperty(UIA_NamePropertyId);
        cacheReq->AddProperty(UIA_AutomationIdPropertyId); // ID
        cacheReq->AddProperty(UIA_ClassNamePropertyId);
        cacheReq->AddProperty(UIA_ControlTypePropertyId);
        cacheReq->AddProperty(UIA_BoundingRectanglePropertyId);
        cacheReq->AddProperty(UIA_IsOffscreenPropertyId);
        cacheReq->AddProperty(UIA_IsEnabledPropertyId);
        
        // VALUE (Metin Değeri)
        cacheReq->AddProperty(UIA_ValueValuePropertyId); 

        // Patterns (Hızlı Erişim)
        cacheReq->AddPattern(UIA_ValuePatternId);
        cacheReq->AddPattern(UIA_InvokePatternId);
        cacheReq->AddPattern(UIA_TogglePatternId);
        cacheReq->AddPattern(UIA_SelectionItemPatternId);
        cacheReq->AddPattern(UIA_ScrollPatternId);

        cacheReq->put_TreeScope(TreeScope_Subtree);
    }

    // 3. Walker Seç (RawView = Her şey, tüm DOM)
    IUIAutomationTreeWalker* walker = nullptr;
    uia_->get_RawViewWalker(&walker); 

    if (walker && cacheReq) {
        // Root'un kendisini de ekleyelim mi? Genelde Window çerçevesidir.
        // RecursiveWalk fonksiyonu çocuklardan başlar, root'u biz manuel ekleyelim.
        UiaElem rootElem{};
        // Root için cache kullanamayız çünkü ElementFromHandle ile aldık (cache yok).
        // Bu yüzden root'u atlayıp direkt içeri dalıyoruz.
        
        RecursiveWalk(walker, root, cacheReq, out, 0, max_elems);
    }

    if (walker) walker->Release();
    if (cacheReq) cacheReq->Release();
    root->Release();

    return out;
}