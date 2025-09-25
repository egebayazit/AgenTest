#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <optional>

struct AR_Point { int x{}, y{}; };
struct AR_Rect  { int x{}, y{}, w{}, h{}; };

enum class AR_MouseButton { Left, Right, Middle };
enum class AR_Mod { Ctrl, Alt, Shift, Win };

struct AR_ClickOpts {
    AR_MouseButton button{AR_MouseButton::Left};
    int  click_count{1};
    std::vector<AR_Mod> modifiers{};
    std::optional<AR_Point> point{};
    std::optional<AR_Rect>  rect{};
};

class ActionRobot {
public:
    ActionRobot();

    bool Click(const AR_ClickOpts& opts);
    bool TypeText(const std::u16string& text, int per_key_delay_ms = 0, bool send_enter = false);
    bool KeyCombo(const std::vector<std::string>& combo);
    bool Drag(const AR_Point& from, const AR_Point& to, AR_MouseButton btn = AR_MouseButton::Left, int hold_ms = 100);
    bool Scroll(int delta, bool horizontal = false, std::optional<AR_Point> at = std::nullopt);
    bool KeyDownByName(const std::string& key);
    bool KeyUpByName(const std::string& key);
    bool Move(const AR_Point& p, int settle_ms = 0);

    const std::string& last_error() const { return last_error_; }

private:
    void set_error(const std::string& s);
    static AR_Point CenterOf(const AR_Rect& r);
    bool MoveCursor(const AR_Point& p);
    void SleepMs(int ms);

    bool ModifiersDown(const std::vector<AR_Mod>& mods);
    void ModifiersUpReverse(const std::vector<AR_Mod>& mods);

    bool MouseDown(AR_MouseButton b);
    bool MouseUp(AR_MouseButton b);
    bool KeyDown(WORD vk);
    bool KeyUp(WORD vk);
    std::optional<WORD> VkFromName(const std::string& name);

private:
    std::string last_error_;
};