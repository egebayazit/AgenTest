#include "../third_party/httplib.h"
#include "action_handler.h"
#include <iostream>

int main(){
    httplib::Server svr;
    action::ActionHandler handler;

    svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res){
        res.set_content(R"({"status":"ok"})","application/json");
    });

    svr.Post("/action", [&handler](const httplib::Request& req, httplib::Response& res){
        if(req.get_header_value("Content-Type").find("application/json")==std::string::npos){
            res.status=400;
            res.set_content(R"({"status":"error","code":"INVALID_CONTENT_TYPE"})","application/json");
            return;
        }
        res.set_content(handler.Handle(req.body), "application/json");
    });

    std::cout << "Action server: http://127.0.0.1:18080\n";
    svr.listen("0.0.0.0", 18080);
    return 0;
}
