#include "action_robot.h"
#include <chrono>
#include <thread>
#include <unordered_map>
#include <algorithm>

#pragma comment(lib, "user32.lib")

static DWORD btnDownFlag(AR_MouseButton b){
    switch(b){ case AR_MouseButton::Left: return MOUSEEVENTF_LEFTDOWN;
               case AR_MouseButton::Right: return MOUSEEVENTF_RIGHTDOWN;
               case AR_MouseButton::Middle: return MOUSEEVENTF_MIDDLEDOWN; }
    return MOUSEEVENTF_LEFTDOWN;
}
static DWORD btnUpFlag(AR_MouseButton b){
    switch(b){ case AR_MouseButton::Left: return MOUSEEVENTF_LEFTUP;
               case AR_MouseButton::Right: return MOUSEEVENTF_RIGHTUP;
               case AR_MouseButton::Middle: return MOUSEEVENTF_MIDDLEUP; }
    return MOUSEEVENTF_LEFTUP;
}

ActionRobot::ActionRobot(){
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
}
void ActionRobot::set_error(const std::string& s){ last_error_ = s; }
void ActionRobot::SleepMs(int ms){ if(ms>0) std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
AR_Point ActionRobot::CenterOf(const AR_Rect& r){ return AR_Point{ r.x + r.w/2, r.y + r.h/2 }; }

bool ActionRobot::MoveCursor(const AR_Point& p){
    if(!SetCursorPos(p.x, p.y)){ set_error("SetCursorPos failed: " + std::to_string(GetLastError())); return false; }
    return true;
}

bool ActionRobot::MouseDown(AR_MouseButton b){
    INPUT in{}; in.type=INPUT_MOUSE; in.mi.dwFlags=btnDownFlag(b);
    if(SendInput(1,&in,sizeof(INPUT))!=1){ set_error("MouseDown SendInput failed"); return false; }
    return true;
}
bool ActionRobot::MouseUp(AR_MouseButton b){
    INPUT in{}; in.type=INPUT_MOUSE; in.mi.dwFlags=btnUpFlag(b);
    if(SendInput(1,&in,sizeof(INPUT))!=1){ set_error("MouseUp SendInput failed"); return false; }
    return true;
}
bool ActionRobot::KeyDown(WORD vk){
    INPUT in{}; in.type=INPUT_KEYBOARD; in.ki.wVk=vk;
    if(SendInput(1,&in,sizeof(INPUT))!=1){ set_error("KeyDown failed"); return false; }
    return true;
}
bool ActionRobot::KeyUp(WORD vk){
    INPUT in{}; in.type=INPUT_KEYBOARD; in.ki.wVk=vk; in.ki.dwFlags=KEYEVENTF_KEYUP;
    if(SendInput(1,&in,sizeof(INPUT))!=1){ set_error("KeyUp failed"); return false; }
    return true;
}

bool ActionRobot::ModifiersDown(const std::vector<AR_Mod>& mods){
    for(auto m:mods){
        WORD vk=0;
        switch(m){ case AR_Mod::Ctrl: vk=VK_CONTROL; break;
                   case AR_Mod::Alt:  vk=VK_MENU;    break;
                   case AR_Mod::Shift:vk=VK_SHIFT;   break;
                   case AR_Mod::Win:  vk=VK_LWIN;    break; }
        if(!KeyDown(vk)) return false;
    }
    return true;
}
void ActionRobot::ModifiersUpReverse(const std::vector<AR_Mod>& mods){
    for(auto it=mods.rbegin(); it!=mods.rend(); ++it){
        WORD vk=0;
        switch(*it){ case AR_Mod::Ctrl: vk=VK_CONTROL; break;
                     case AR_Mod::Alt:  vk=VK_MENU;    break;
                     case AR_Mod::Shift:vk=VK_SHIFT;   break;
                     case AR_Mod::Win:  vk=VK_LWIN;    break; }
        KeyUp(vk);
    }
}

std::optional<WORD> ActionRobot::VkFromName(const std::string& name){
    static const std::unordered_map<std::string, WORD> map = {
        {"ctrl",VK_CONTROL},{"alt",VK_MENU},{"shift",VK_SHIFT},{"win",VK_LWIN},
        {"enter",VK_RETURN},{"esc",VK_ESCAPE},{"tab",VK_TAB},{"space",VK_SPACE},
        {"backspace",VK_BACK},{"delete",VK_DELETE},{"home",VK_HOME},{"end",VK_END},
        {"pgup",VK_PRIOR},{"pgdn",VK_NEXT},{"left",VK_LEFT},{"right",VK_RIGHT},
        {"up",VK_UP},{"down",VK_DOWN},
        {"f1",VK_F1},{"f2",VK_F2},{"f3",VK_F3},{"f4",VK_F4},{"f5",VK_F5},
        {"f6",VK_F6},{"f7",VK_F7},{"f8",VK_F8},{"f9",VK_F9},{"f10",VK_F10},
        {"f11",VK_F11},{"f12",VK_F12}
    };
    auto lower = name; std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if(map.count(lower)) return map.at(lower);
    if(lower.size()==1){
        HKL layout=GetKeyboardLayout(0);
        SHORT s=VkKeyScanExW((WCHAR)lower[0], layout);
        if(s==-1) return std::nullopt;
        return LOBYTE(s);
    }
    return std::nullopt;
}

bool ActionRobot::Click(const AR_ClickOpts& opts){
    AR_Point p{};
    if(opts.point) p=*opts.point;
    else if(opts.rect) p=CenterOf(*opts.rect);
    else { set_error("Click: no target"); return false; }

    if(!MoveCursor(p)) return false;
    SleepMs(20);

    if(!ModifiersDown(opts.modifiers)) return false;
    for(int i=0;i<(std::max)(1,opts.click_count);++i){
        if(!MouseDown(opts.button)){ ModifiersUpReverse(opts.modifiers); return false; }
        SleepMs(50);
        if(!MouseUp(opts.button)){ ModifiersUpReverse(opts.modifiers); return false; }
        if(opts.click_count==2) SleepMs(120);
    }
    ModifiersUpReverse(opts.modifiers);
    return true;
}

bool ActionRobot::TypeText(const std::u16string& text, int per_key_delay_ms, bool send_enter){
    for(char16_t ch : text){
        INPUT down{}; down.type=INPUT_KEYBOARD; down.ki.dwFlags=KEYEVENTF_UNICODE; down.ki.wScan=ch;
        INPUT up{};   up.type=INPUT_KEYBOARD;   up.ki.dwFlags=KEYEVENTF_UNICODE|KEYEVENTF_KEYUP; up.ki.wScan=ch;
        INPUT arr[2]{down,up};
        if(SendInput(2,arr,sizeof(INPUT))!=2){ set_error("TypeText SendInput failed"); return false; }
        SleepMs(per_key_delay_ms);
    }
    if(send_enter){ if(!KeyDown(VK_RETURN) || !KeyUp(VK_RETURN)){ set_error("Enter failed"); return false; } }
    return true;
}

bool ActionRobot::KeyCombo(const std::vector<std::string>& combo){
    std::vector<WORD> vks; vks.reserve(combo.size());
    for(auto& k:combo){ auto vk=VkFromName(k); if(!vk){ set_error("Unknown key: "+k); return false; } vks.push_back(*vk); }
    for(auto vk:vks) if(!KeyDown(vk)){ set_error("KeyDown failed"); return false; }
    for(auto it=vks.rbegin(); it!=vks.rend(); ++it) if(!KeyUp(*it)){ set_error("KeyUp failed"); return false; }
    return true;
}

bool ActionRobot::Drag(const AR_Point& from, const AR_Point& to, AR_MouseButton btn, int hold_ms){
    if(!MoveCursor(from)) return false;
    SleepMs(20);
    if(!MouseDown(btn)) return false;
    SleepMs(hold_ms);
    if(!MoveCursor(to)){ MouseUp(btn); return false; }
    SleepMs(20);
    if(!MouseUp(btn)) return false;
    return true;
}

bool ActionRobot::Scroll(int delta, bool horizontal, std::optional<AR_Point> at){
    if(at && !MoveCursor(*at)) return false;
    INPUT in{}; in.type=INPUT_MOUSE;
    if(horizontal){ in.mi.dwFlags=MOUSEEVENTF_HWHEEL; in.mi.mouseData=delta; }
    else          { in.mi.dwFlags=MOUSEEVENTF_WHEEL;  in.mi.mouseData=delta; }
    if(SendInput(1,&in,sizeof(INPUT))!=1){ set_error("Scroll SendInput failed"); return false; }
    return true;
}

bool ActionRobot::KeyDownByName(const std::string& key){ auto vk=VkFromName(key); if(!vk){ set_error("Unknown key"); return false; } return KeyDown(*vk); }
bool ActionRobot::KeyUpByName(const std::string& key){ auto vk=VkFromName(key); if(!vk){ set_error("Unknown key"); return false; } return KeyUp(*vk); }
bool ActionRobot::Move(const AR_Point& p, int settle_ms){ if(!MoveCursor(p)) return false; SleepMs(settle_ms); return true; }