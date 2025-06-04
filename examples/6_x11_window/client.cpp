#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <cstring>
#include "def.h"

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

void send_binding(Window window_id, pid_t pid) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "Socket creation error" << std::endl;
        return;
    }
    
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080);
    
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        return;
    }
    
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection Failed" << std::endl;
        return;
    }
    
    std::string message = "BIND " + std::to_string(window_id) + " " + std::to_string(pid);
    send(sock, message.c_str(), message.length(), 0);
    close(sock);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [args...]" << std::endl;
        return 1;
    }
    
    // Get the active window
    Display* display = XOpenDisplay(nullptr);
    if (!display) {
        std::cerr << "Failed to open X display" << std::endl;
        return 1;
    }
    
    Window active_window = get_active_window(display);
    if (!active_window) {
        std::cerr << "Failed to get active window" << std::endl;
        XCloseDisplay(display);
        return 1;
    }
    
    XCloseDisplay(display);

    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        execvp(argv[1], &argv[1]);
        std::cerr << "Failed to execute command" << std::endl;
        exit(EXIT_FAILURE);
    } else if (pid > 0) {
        // Parent process
        send_binding(active_window, pid);
        
        int status;
        waitpid(pid, &status, 0);
        return WEXITSTATUS(status);
    } else {
        std::cerr << "Fork failed" << std::endl;
        return 1;
    }
}