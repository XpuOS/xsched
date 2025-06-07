#include <unistd.h>
#include <CLI/CLI.hpp>

#include "launcher.h"
#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/protocol/def.h"

using namespace xsched::service;

int main(int argc, char **argv)
{
    auto app = std::make_unique<CLI::App>("X11 Launcher");
    std::string addr = "127.0.0.1";
    app->add_option("-a,--addr", addr, "X11 monitor ipv4 address. Default is [127.0.0.1].")
       ->check(CLI::ValidIPV4);
    uint16_t port = XSCHED_X11_MONITOR_DEFAULT_PORT;
    app->add_option("-l,--listen", port, "X11 monitor port. Default is ["
                    TOSTRING(XSCHED_X11_MONITOR_DEFAULT_PORT) "].")->check(CLI::Range(0x1U, 0xFFFFU));
    app->allow_extras();
    CLI11_PARSE(*app, argc, argv);
    std::vector<std::string> extra = app->remaining();

    X11Launcher launcher(addr, port);
    launcher.SetBind();

    size_t dash_pos = 0;
    for (; dash_pos < extra.size(); ++dash_pos) {
        if (extra[dash_pos] == "--") break;
    }
    if (dash_pos >= extra.size()) {
        XWARN("please use -- to separate");
        return -1;
    }
    if (dash_pos + 1 >= extra.size()) {
        XWARN("no command provided, exit");
        return -1;
    }

    std::string cmd;
    std::vector<char *> args;
    for (size_t i = dash_pos + 1; i < extra.size(); ++i) {
        cmd += extra[i] + " ";
        args.push_back(extra[i].data());
    }
    args.push_back(nullptr);
    cmd.pop_back();

    // launch
    execvp(extra[dash_pos + 1].c_str(), args.data());
    std::string err = "command \"" + cmd + "\" failed";
    perror(err.c_str());
    return -1;
}
