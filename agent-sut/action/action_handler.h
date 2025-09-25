#pragma once
#include <string>
#include "action_robot.h"

namespace action {

class ActionHandler {
public:
    ActionHandler();
    // JSON body (string) alır, eylemleri uygular, JSON string döner (ACK-only)
    std::string Handle(const std::string& json_body);

private:
    ActionRobot robot_;
};

} // namespace action