#include "monitor.h"
#include "xsched/utils/log.h"
#include "xsched/protocol/def.h"

#ifdef CPPHTTPLIB_THREAD_POOL_COUNT
#undef CPPHTTPLIB_THREAD_POOL_COUNT
#define CPPHTTPLIB_THREAD_POOL_COUNT 1
#endif
#include <httplib.h>
#include <CLI/CLI.hpp>

using namespace xsched::service;

int main(int argc, char **argv)
{
    auto app = std::make_unique<CLI::App>("X11 Monitor");
    std::string addr = "127.0.0.1";
    app->add_option("-a,--addr", addr, "XSched server ipv4 address. Default is [127.0.0.1].")
       ->check(CLI::ValidIPV4);
    uint16_t xserver_port = XSCHED_SERVER_DEFAULT_PORT;
    app->add_option("-p,--port", xserver_port, "XSched server port. Default is ["
                    TOSTRING(XSCHED_SERVER_DEFAULT_PORT) "].")->check(CLI::Range(0x1U, 0xFFFFU));
    uint16_t listen_port = XSCHED_X11_MONITOR_DEFAULT_PORT;
    app->add_option("-l,--listen", listen_port, "X11 monitor listen port. Default is ["
                    TOSTRING(XSCHED_X11_MONITOR_DEFAULT_PORT) "].")->check(CLI::Range(0x1U, 0xFFFFU));
    CLI11_PARSE(*app, argc, argv);

    X11Monitor monitor(addr, xserver_port);
    monitor.ListenAllSessions();

    auto post_bind = [&](const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &) {
        std::string display = ":";
        if (!req.has_param("display")) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"missing display\"}", "application/json");
            return;
        }
        display += req.get_param_value("display");

        Window window = 0;
        if (req.has_param("window")) {
            size_t window_pos = 0;
            std::string window_str = req.get_param_value("window");
            window = std::stoll(window_str, &window_pos, 10);
            if (window_pos != window_str.length()) {
                res.status = httplib::StatusCode::BadRequest_400;
                res.set_content("{\"error\": \"invalid window\"}", "application/json");
                return;
            }
        }

        PID pid = 0;
        if (req.has_param("pid")) {
            size_t pid_pos = 0;
            std::string pid_str = req.get_param_value("pid");
            pid = std::stoll(pid_str, &pid_pos, 10);
            if (pid_pos != pid_str.length()) {
                res.status = httplib::StatusCode::BadRequest_400;
                res.set_content("{\"error\": \"invalid pid\"}", "application/json");
                return;
            }
        }

        monitor.SetWindowPID(display, window, pid);
        res.status = httplib::StatusCode::OK_200;
        res.set_content("{\"info\": \"success\"}", "application/json");
    };

    XINFO("x11 monitor listen on port %d", listen_port);
    httplib::Server http_server;
    // request = "/bind?display=0&window=5678&pid=1234"
    // NOTICE: for the display parameter, ":" is omitted, so the display is ":0"
    http_server.Post("/bind", post_bind);
    http_server.listen("0.0.0.0", listen_port);
    monitor.Join();
    return 0;
}
