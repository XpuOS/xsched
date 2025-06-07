#include <memory>
#include <vector>
#include <thread>
#include <dirent.h>

#include "monitor.h"
#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"

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

X11Monitor::X11Monitor(const std::string &addr, uint16_t port): client_(addr, port)
{
    XINFO("xserver address: %s, %d", addr.c_str(), port);
    XSetErrorHandler(X11ErrorHandler);
}

void X11Monitor::Join()
{
    for (auto &thread : threads_) thread.join();
}

void X11Monitor::ListenAllSessions()
{
    DIR *dir = opendir("/tmp/.X11-unix");
    if (!dir) XERRO("failed to open /tmp/.X11-unix");

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == 'X' && isdigit(entry->d_name[1])) {
            // file name: X0, X1, ...
            std::string display_name = ":";
            uint64_t display_id = atol(entry->d_name + 1);
            display_name += std::to_string(display_id);
            Display *display = XOpenDisplay(display_name.c_str());
            if (!display) {
                XWARN("skip display %s", display_name.c_str());
                continue;
            }
            auto x11_display = std::make_shared<X11Display>();
            x11_display->display = display;
            x11_display->name = display_name;
            x11_display->id = display_id;
            displays_[display_name] = x11_display;
            threads_.emplace_back(&X11Monitor::ListenWindowActive, this, x11_display);
        }
    }
    closedir(dir);
}

void X11Monitor::ListenWindowActive(std::shared_ptr<X11Display> x11_display)
{
    Display *display = x11_display->display;
    uint64_t display_id = x11_display->id;
    XINFO("listening display %s", x11_display->name.c_str());

    PID active_pid = 0;
    Window root = DefaultRootWindow(display);
    Atom net_active_window = XInternAtom(display, "_NET_ACTIVE_WINDOW", True);
    XASSERT(net_active_window != None, "failed to get _NET_ACTIVE_WINDOW atom");

    // get current active window
    Atom actual_type;
    int actual_format;
    unsigned long nitems, bytes_after;
    unsigned char *prop = nullptr;
    int status = XGetWindowProperty(display, root, net_active_window, 0, (~0L), False,
                                    AnyPropertyType, &actual_type, &actual_format,
                                    &nitems, &bytes_after, &prop);
    if (status == Success && prop && nitems) {
        Window active_window = *(Window*)prop;
        active_pid = GetWindowPID(x11_display, active_window);
        if (active_pid != 0) {
            client_.SetActiveWindow(active_pid, display_id);
            XINFO("display %s active window: 0x%lx, pid: %d",
                  x11_display->name.c_str(), active_window, active_pid);
        }
    }
    if (prop) {
        XFree(prop);
        prop = nullptr;
    }

    // set up active window listener
    XSelectInput(display, root, PropertyChangeMask);
    while (true) {
        XEvent ev;
        if (XNextEvent(display, &ev) != Success) break;
        if (ev.type == PropertyNotify && ev.xproperty.atom == net_active_window) {
            status = XGetWindowProperty(display, root, net_active_window, 0, (~0L), False,
                                        AnyPropertyType, &actual_type, &actual_format,
                                        &nitems, &bytes_after, &prop);
            if (status == Success && prop && nitems) {
                Window active_window = *(Window*)prop;
                active_pid = GetWindowPID(x11_display, active_window);
                if (active_pid != 0) {
                    client_.SetActiveWindow(active_pid, display_id);
                    XINFO("display %s active window: 0x%lx, pid: %d",
                          x11_display->name.c_str(), active_window, active_pid);
                }
            }
            if (prop) {
                XFree(prop);
                prop = nullptr;
            }
        } else if (ev.type == DestroyNotify) {
            Window win = ev.xdestroywindow.window;
            std::lock_guard<std::mutex> lock(x11_display->mtx);
            if (x11_display->pids.count(win)) {
                XINFO("display %s window 0x%lx destroyed, remove it from the bind map",
                      x11_display->name.c_str(), win);
                x11_display->pids.erase(win);
            }
        }
    }
    XCloseDisplay(display);
}

void X11Monitor::SetWindowPID(const std::string &display_name, Window window, PID pid)
{
    XINFO("display %s window 0x%lx bind pid %d", display_name.c_str(), window, pid);
    auto it = displays_.find(display_name);
    if (it == displays_.end()) {
        XWARN("display %s not opened", display_name.c_str());
        return;
    }

    // listen the window destroy notify
    XSelectInput(it->second->display, window, StructureNotifyMask);

    // check if already destoryed by getting its pid
    Atom actual_type;
    int actual_format;
    unsigned long nitems, bytes_after;
    unsigned char* prop_pid = nullptr;
    Atom pid_atom = XInternAtom(it->second->display, "_NET_WM_PID", True);
    XASSERT(pid_atom != None, "failed to get _NET_WM_PID atom");
    int status = XGetWindowProperty(it->second->display, window, pid_atom, 0, 1, False,
                                    XA_CARDINAL, &actual_type, &actual_format,
                                    &nitems, &bytes_after, &prop_pid);
    if (prop_pid) XFree(prop_pid);
    if (status != Success) {
        XWARN("display %s window 0x%lx already destoryed", display_name.c_str(), window);
        return;
    }

    it->second->mtx.lock();
    it->second->pids[window] = pid;
    it->second->mtx.unlock();

    // set the active window
    client_.SetActiveWindow(pid, it->second->id);
}

PID X11Monitor::GetWindowPID(std::shared_ptr<X11Display> x11_display, Window window)
{
    {
        // search in the bind map first
        std::lock_guard<std::mutex> lock(x11_display->mtx);
        auto it = x11_display->pids.find(window);
        if (it != x11_display->pids.end()) return it->second;
    }

    // get the pid of the window by X API
    pid_t pid = 0;
    Display *display = x11_display->display;
    Atom pid_atom = XInternAtom(display, "_NET_WM_PID", True);
    XASSERT(pid_atom != None, "failed to get _NET_WM_PID atom");

    Atom actual_type;
    int actual_format;
    unsigned long nitems, bytes_after;
    unsigned char* prop_pid = nullptr;
    int status = XGetWindowProperty(display, window, pid_atom, 0, 1, False,
                                    XA_CARDINAL, &actual_type, &actual_format,
                                    &nitems, &bytes_after, &prop_pid);
    if (status == Success && prop_pid && nitems) {
        pid = *(pid_t*)prop_pid;
    }
    if (prop_pid) XFree(prop_pid);

    return (PID)pid;
}
