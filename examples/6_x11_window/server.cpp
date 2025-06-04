#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XInput2.h>
#include <thread>
#include <mutex>
#include <cstring>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <algorithm>
#include "def.h"

std::mutex bindings_mutex;
std::unordered_map<Window, pid_t> window_pid_bindings;

pid_t get_window_pid(Display* display, Window window) {
    Atom actual_type;
    int actual_format;
    unsigned long nitems, bytes_after;
    unsigned char* data = nullptr;
    pid_t pid = 0;

    Atom prop = XInternAtom(display, "_NET_WM_PID", True);
    if (prop == None) {
        return 0;
    }

    if (XGetWindowProperty(display, window, prop, 0, 1, False, XA_CARDINAL,
                          &actual_type, &actual_format, &nitems, &bytes_after,
                          &data) == Success && data) {
        if (actual_type == XA_CARDINAL && actual_format == 32 && nitems == 1) {
            pid = *reinterpret_cast<pid_t*>(data);
        }
        XFree(data);
    }

    return pid;
}

std::string get_process_command(pid_t pid) {
    std::string cmdline_path = "/proc/" + std::to_string(pid) + "/cmdline";
    FILE* cmdline_file = fopen(cmdline_path.c_str(), "r");
    if (!cmdline_file) {
        return "";
    }

    std::string command;
    char buffer[1024];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), cmdline_file)) > 0) {
        for (size_t i = 0; i < bytes_read; ++i) {
            if (buffer[i] == '\0') {
                buffer[i] = ' ';
            }
        }
        command.append(buffer, bytes_read);
    }
    fclose(cmdline_file);

    if (!command.empty() && command.back() == ' ') {
        command.pop_back();
    }

    return command;
}

Window get_active_window(Display* display) {
    Window active_window = 0;
    Atom actual_type;
    int actual_format;
    unsigned long nitems, bytes_after;
    unsigned char* data = nullptr;

    Atom prop = XInternAtom(display, "_NET_ACTIVE_WINDOW", True);
    if (prop == None) {
        return 0;
    }

    if (XGetWindowProperty(display, XDefaultRootWindow(display), prop, 0, 1, False, XA_WINDOW,
                          &actual_type, &actual_format, &nitems, &bytes_after,
                          &data) == Success && data) {
        if (actual_type == XA_WINDOW && actual_format == 32 && nitems == 1) {
            active_window = *reinterpret_cast<Window*>(data);
        }
        XFree(data);
    }

    return active_window;
}

std::string get_window_title(Display* display, Window window) {
    XTextProperty text_prop;
    std::string title;
    
    if (XGetWMName(display, window, &text_prop) && text_prop.value && text_prop.nitems > 0) {
        int count = 0;
        char** list = nullptr;
        if (Xutf8TextPropertyToTextList(display, &text_prop, &list, &count) >= Success && count > 0 && list) {
            title = list[0];
            XFreeStringList(list);
        }
        XFree(text_prop.value);
    }
    
    return title;
}

void monitor_active_window(Display* display) {
    Window last_active_window = 0;
    
    XEvent event;
    Atom net_active_window = XInternAtom(display, "_NET_ACTIVE_WINDOW", False);
    XSelectInput(display, XDefaultRootWindow(display), PropertyChangeMask);
    
    while (true) {
        XNextEvent(display, &event);
        if (event.type == PropertyNotify && event.xproperty.atom == net_active_window) {
            Window active_window = get_active_window(display);
            if (active_window && active_window != last_active_window) {
                last_active_window = active_window;
                
                pid_t pid = get_window_pid(display, active_window);
                
                std::lock_guard<std::mutex> lock(bindings_mutex);
                auto it = window_pid_bindings.find(active_window);
                if (it != window_pid_bindings.end()) {
                    pid = it->second;
                }
                
                std::string command = pid ? get_process_command(pid) : "Unknown";
                std::string title = get_window_title(display, active_window);
                
                std::cout << "Active Window Changed:" << std::endl;
                std::cout << "  Window ID: " << active_window << std::endl;
                std::cout << "  PID: " << pid << std::endl;
                std::cout << "  Command: " << command << std::endl;
                std::cout << "  Title: " << title << std::endl;
            }
        }
    }
}

void handle_client(int client_socket) {
    char buffer[1024] = {0};
    read(client_socket, buffer, sizeof(buffer));
    
    std::istringstream iss(buffer);
    std::string command;
    Window window_id;
    pid_t pid;
    
    iss >> command;
    if (command == "BIND") {
        iss >> window_id >> pid;
        if (iss) {
            std::lock_guard<std::mutex> lock(bindings_mutex);
            window_pid_bindings[window_id] = pid;
            std::cout << "Bound Window " << window_id << " to PID " << pid << std::endl;
        }
    }
    
    close(client_socket);
}

void start_server() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);
    
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Server listening on port 8080..." << std::endl;
    
    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }
        
        std::thread(handle_client, new_socket).detach();
    }
}

int main() {
    Display* display = XOpenDisplay(nullptr);
    if (!display) {
        std::cerr << "Failed to open X display" << std::endl;
        return 1;
    }

    std::thread monitor_thread(monitor_active_window, display);
    std::thread server_thread(start_server);
    
    monitor_thread.join();
    server_thread.join();
    
    XCloseDisplay(display);
    return 0;
}