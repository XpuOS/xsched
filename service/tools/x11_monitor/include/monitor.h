#pragma once

#include <memory>
#include <unordered_map>

#include "client.h"
#include "xsched/utils/common.h"

// forward def to avoid include X11 in headers
typedef unsigned long XID;
typedef XID Window;
typedef struct _XDisplay Display;

namespace xsched::service
{

struct X11Display
{
    Display *display;
    std::string name;
    uint64_t id;
    std::mutex mtx;
    std::unordered_map<Window, PID> pids;
};

class X11Monitor
{
public:
    X11Monitor(const std::string &addr, uint16_t port);
    ~X11Monitor() = default;

    void Join();
    void ListenAllSessions();
    void ListenWindowActive(std::shared_ptr<X11Display> x11_display);

    void SetWindowPID(const std::string &display_name, Window window, PID pid);
    PID GetWindowPID(std::shared_ptr<X11Display> x11_display, Window window);

private:
    Client client_;
    std::vector<std::thread> threads_;
    std::unordered_map<std::string, std::shared_ptr<X11Display>> displays_;
};

} // namespace xsched::service
