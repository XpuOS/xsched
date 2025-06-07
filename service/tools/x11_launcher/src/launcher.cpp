#include "launcher.h"
#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"

#include <httplib.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>

using namespace xsched::service;

int X11ErrorHandler(Display *display, XErrorEvent *error)
{
    char err_str[1024];
    XGetErrorText(display, error->error_code, err_str, sizeof(err_str));
    XDEBG("X11 error %d: %s", error->error_code, err_str);
    return 0;
}

X11Launcher::X11Launcher(const std::string &addr, uint16_t port): kAddr(addr), kPort(port)
{
    XINFO("xserver address: %s, %d", addr.c_str(), port);
    XSetErrorHandler(X11ErrorHandler);
}

void X11Launcher::SetBind()
{
    Display *display = XOpenDisplay(nullptr);
    XASSERT(display != nullptr, "failed to open display, please use in a terminal window");
    std::string display_name(XDisplayString(display));
    XASSERT(display_name.length() > 0 && display_name[0] == ':',
            "illegal display name: %s", display_name.c_str());
    display_name = display_name.substr(1);

    // get current active window
    Window active_window = 0;
    Window root = DefaultRootWindow(display);
    Atom net_active_window = XInternAtom(display, "_NET_ACTIVE_WINDOW", True);
    XASSERT(net_active_window != None, "failed to get _NET_ACTIVE_WINDOW atom");

    Atom actual_type;
    int actual_format;
    unsigned long nitems, bytes_after;
    unsigned char *prop = nullptr;
    int status = XGetWindowProperty(display, root, net_active_window, 0, (~0L), False,
                                    AnyPropertyType, &actual_type, &actual_format,
                                    &nitems, &bytes_after, &prop);
    XASSERT(status == Success && prop && nitems, "failed to get active window");
    active_window = *(Window*)prop;
    XFree(prop);

    // get current pid
    PID pid = GetProcessId();

#undef Success
    httplib::Client client(kAddr, kPort);
    std::string url = "/bind?display=" + display_name
                    + "&window=" + std::to_string(active_window)
                    + "&pid=" + std::to_string(pid);
    httplib::Result res = client.Post(url);
    XASSERT(res.error() == httplib::Error::Success && res != nullptr,
            "failed to get response, error: %s", httplib::to_string(res.error()).c_str())
    XASSERT(res->status == httplib::StatusCode::OK_200,
            "failed to post pid bind, response code: %d, message: %s",
            res->status, res->body.c_str())
    XINFO("bind pid response: %s", res->body.c_str());
}
