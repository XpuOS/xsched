#include <memory>
#include <csignal>

#include "server.h"
#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/protocol/def.h"
#include "xsched/protocol/names.h"

using namespace xsched::service;
using namespace xsched::protocol;

static std::unique_ptr<Server> server = nullptr;

void ExitSignal(int)
{
    if (server) server->Stop();
}

int main(int argc, char *argv[])
{
    std::string policy_name = GetPolicyTypeName(kPolicyHighestPriorityFirst);
    std::string port = std::to_string(XSCHED_SERVER_DEFAULT_PORT);

    if (argc == 1) {
        XINFO("using default: %s %s %s", argv[0], policy_name.c_str(), port.c_str());
    } else if (argc == 3) {
        policy_name = argv[1];
        port = argv[2];
    } else {
        XINFO("Usage: %s <policy name> <server port>", argv[0]);
        XINFO("default: %s %s %s", argv[0], policy_name.c_str(), port.c_str());
        XERRO("invalid arguments, abort...");
    }

    XASSERT(signal(SIGINT, ExitSignal) != SIG_ERR, "failed to set SIGINT handler");
#if defined(__linux__)
    XASSERT(signal(SIGQUIT, ExitSignal) != SIG_ERR, "failed to set SIGQUIT handler");
#endif

    server = std::make_unique<Server>(policy_name, port);
    server->Run();
    return 0;
}
