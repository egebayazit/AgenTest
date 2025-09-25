#include "action_handler.h"
#include "third_party/json.hpp"
#include <algorithm>
#include <chrono>
#include <thread>
#include <Windows.h>

// UTF-8 -> UTF-16 (char16_t) dönüşümü
static std::u16string Utf8ToU16(const std::string& s) {
    if (s.empty()) return {};
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (wlen <= 0) return {};
    std::wstring w; w.resize(wlen);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], wlen);
    return std::u16string(w.begin(), w.end()); // wchar_t (Win=16-bit) -> char16_t
}


using json = nlohmann::json;
using namespace action;

static AR_MouseButton parseButton(const std::string& s){
    if(s=="right") return AR_MouseButton::Right;
    if(s=="middle") return AR_MouseButton::Middle;
    return AR_MouseButton::Left;
}
static std::vector<AR_Mod> parseMods(const json& jmods){
    std::vector<AR_Mod> out;
    for(auto& m : jmods){
        std::string s = m.get<std::string>(); std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        if(s=="ctrl") out.push_back(AR_Mod::Ctrl);
        else if(s=="alt") out.push_back(AR_Mod::Alt);
        else if(s=="shift") out.push_back(AR_Mod::Shift);
        else if(s=="win") out.push_back(AR_Mod::Win);
    }
    return out;
}

ActionHandler::ActionHandler() = default;

std::string ActionHandler::Handle(const std::string& body){
    json req;
    try { req = json::parse(body); }
    catch (...) {
        return json{{"status","error"},{"code","INVALID_PAYLOAD"},{"detail","JSON parse failed"}}.dump();
    }

    const std::string action_id = req.value("action_id","");
    if(!req.contains("steps") || !req["steps"].is_array()){
        return json{{"status","error"},{"action_id",action_id},
                    {"code","INVALID_PAYLOAD"},{"detail","steps[] missing"}}.dump();
    }

    int applied = 0;
    for(auto& step : req["steps"]){
        if(!step.contains("type")){
            return json{{"status","error"},{"action_id",action_id},
                        {"code","INVALID_PAYLOAD"},{"detail","step.type missing"},{"applied",applied}}.dump();
        }
        const std::string type = step["type"].get<std::string>();

        if(type=="click"){
            AR_ClickOpts opts;
            if(step.contains("button"))      opts.button = parseButton(step["button"].get<std::string>());
            if(step.contains("click_count")) opts.click_count = std::max<int>(1, step["click_count"].get<int>());
            if(step.contains("modifiers"))   opts.modifiers = parseMods(step["modifiers"]);

            if(step.contains("target") && step["target"].is_object()){
                auto& t = step["target"];
                if(t.contains("point")){
                    auto p=t["point"]; opts.point = AR_Point{ p.value("x",0), p.value("y",0) };
                } else if(t.contains("rect")){
                    auto r=t["rect"];  opts.rect  = AR_Rect{ r.value("x",0), r.value("y",0), r.value("w",0), r.value("h",0) };
                }
            }
            if(!robot_.Click(opts)){
                return json{{"status","error"},{"action_id",action_id},
                            {"code","INPUT_INJECTION_FAILED"},{"detail",robot_.last_error()},
                            {"applied",applied}}.dump();
            }
        }
        else if (type == "type") {
            const std::string utf8 = step.value("text", std::string{});
            const int delay = step.value("delay_ms", 0);
            const bool enter = step.value("enter", false);

            const std::u16string u16 = Utf8ToU16(utf8);
            if (!robot_.TypeText(u16, delay, enter)) {
            return json{{"status","error"},{"action_id",action_id},
                    {"code","INPUT_INJECTION_FAILED"},{"detail",robot_.last_error()},
                    {"applied",applied}}.dump();
            }
        }

        else if(type=="key_combo"){
            if(!step.contains("combo") || !step["combo"].is_array()){
                return json{{"status","error"},{"action_id",action_id},
                            {"code","INVALID_PAYLOAD"},{"detail","combo missing"},
                            {"applied",applied}}.dump();
            }
            std::vector<std::string> combo;
            for(auto& k: step["combo"]) combo.push_back(k.get<std::string>());
            if(!robot_.KeyCombo(combo)){
                return json{{"status","error"},{"action_id",action_id},
                            {"code","INPUT_INJECTION_FAILED"},{"detail",robot_.last_error()},
                            {"applied",applied}}.dump();
            }
        }
        else if(type=="drag"){
            auto f=step["from"]; auto t=step["to"];
            AR_Point from{ f.value("x",0), f.value("y",0) };
            AR_Point to  { t.value("x",0), t.value("y",0) };
            AR_MouseButton btn = parseButton(step.value("button","left"));
            int hold_ms = step.value("hold_ms",100);
            if(!robot_.Drag(from,to,btn,hold_ms)){
                return json{{"status","error"},{"action_id",action_id},
                            {"code","INPUT_INJECTION_FAILED"},{"detail",robot_.last_error()},
                            {"applied",applied}}.dump();
            }
        }
        else if(type=="scroll"){
            int delta = step.value("delta",0);
            bool horiz = step.value("horizontal",false);
            std::optional<AR_Point> at;
            if(step.contains("at")){ auto a=step["at"]; at = AR_Point{ a.value("x",0), a.value("y",0) }; }
            if(!robot_.Scroll(delta,horiz,at)){
                return json{{"status","error"},{"action_id",action_id},
                            {"code","INPUT_INJECTION_FAILED"},{"detail",robot_.last_error()},
                            {"applied",applied}}.dump();
            }
        }
        else if(type=="key_down"){
            if(!robot_.KeyDownByName(step.value("key",""))){
                return json{{"status","error"},{"action_id",action_id},
                            {"code","INPUT_INJECTION_FAILED"},{"detail",robot_.last_error()},
                            {"applied",applied}}.dump();
            }
        }
        else if(type=="key_up"){
            if(!robot_.KeyUpByName(step.value("key",""))){
                return json{{"status","error"},{"action_id",action_id},
                            {"code","INPUT_INJECTION_FAILED"},{"detail",robot_.last_error()},
                            {"applied",applied}}.dump();
            }
        }
        else if(type=="move" || type=="hover"){
            auto p=step["point"];
            if(!robot_.Move(AR_Point{ p.value("x",0), p.value("y",0) }, step.value("settle_ms",0))){
                return json{{"status","error"},{"action_id",action_id},
                            {"code","INPUT_INJECTION_FAILED"},{"detail",robot_.last_error()},
                            {"applied",applied}}.dump();
            }
        }
        else if(type=="wait"){
            std::this_thread::sleep_for(std::chrono::milliseconds(step.value("ms",0)));
        }
        else {
            return json{{"status","error"},{"action_id",action_id},
                        {"code","UNSUPPORTED_ACTION"},{"detail",type},
                        {"applied",applied}}.dump();
        }
        ++applied;
    }

    json ok{
        {"status","ok"},
        {"action_id",action_id},
        {"timestamp",(long long)std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::system_clock::now().time_since_epoch()).count()},
        {"applied",applied}
    };
    return ok.dump();
}