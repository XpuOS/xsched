#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#endif

#include <algorithm>
#include <unordered_map>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/component/component.hpp>

#include "cli.h"
#include "convert.h"
#include "xsched_assert.h"
#include "xsched/utils/log.h"
#include "xsched/protocol/names.h"
#include "xsched/protocol/device.h"

using namespace ftxui;
using namespace xsched::sched;
using namespace xsched::service;
using namespace xsched::protocol;

Cli::Cli(const std::string &addr, uint16_t port)
{
    client_ = std::make_unique<Client>(addr, port);
}

int Cli::ListXQueues()
{
    std::vector<XQueueStatus> xqueue_status;
    std::unordered_map<PID, std::string> pid_to_cmdline;
    XSCHED_ASSERT(client_->QueryXQueues(xqueue_status, pid_to_cmdline));

    // list the table
    std::vector<std::vector<Element>> rows;

    // header row
    rows.push_back({
        text(" PID ")    | bold | center,
        text(" DEV ")    | bold | center,
        text(" XQUEUE ") | bold | center,
        text(" STAT ")   | bold | center,
        text(" SCHED ")  | bold | center,
        text(" LV ")     | bold | center,
        text(" CMD ")    | bold | center,
    });

    // data rows
    std::sort(xqueue_status.begin(), xqueue_status.end(),
              [](const XQueueStatus &a, const XQueueStatus &b) { return a.pid < b.pid; });
    for (const auto &status : xqueue_status) {
        std::string dev = GetDeviceTypeName(GetDeviceType(status.device))
                        + "(" + ToHex(status.device) + ")";
        auto cmdl_it = pid_to_cmdline.find(status.pid);
        const std::string &cmd = cmdl_it == pid_to_cmdline.end() ? "" : cmdl_it->second;
        auto stat = status.ready ? text(" RDY ") | color(Color::Cyan)
                                 : text(" BLK ") | color(Color::Yellow);
        auto sched = status.suspended ? text(" SUS ") | color(Color::Red)
                                      : text(" RUN ") | color(Color::Green);

        rows.push_back({
            text(" " + std::to_string(status.pid) + " ") | center,
            text(" " + dev + " ") | center,
            text(" " + ToHex(status.handle) + " ") | center,
            stat  | center,
            sched | center,
            text(" " + std::to_string((int)status.level) + " ") | center,
            text(" " + cmd + " ") | center,
        });
    }

    // render the table on terminal
    auto table = gridbox(std::move(rows)) | borderLight;
    auto document = vbox({
        table,
        filler(),
    });

    auto screen = Screen::Create(Dimension::Full(), Dimension::Fit(document));
    Render(screen, document);
    std::cout << screen.ToString() << std::endl;
    return 0;
}

int Cli::TopXQueues(uint64_t interval_ms)
{
    while (true) {
        std::cout << "\033[2J\033[H";
        ListXQueues();
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    return 0;
}

int Cli::ConfigXQueue(XQueueHandle handle, XPreemptLevel level,
                      int64_t threshold, int64_t batch_size)
{
    XSCHED_ASSERT(client_->SetXQueueConfig(handle, level, threshold, batch_size));
    std::cout << "Config of XQueue (" << ToHex(handle) << ") set to level: " << level
              << ", command threshold: " << threshold
              << ", command batch size: " << batch_size << std::endl;
    std::cout << "  Note: 0 for level and -1 for threshold and batch size means no change"
              << std::endl;
    std::cout << "Current XQueue status: " << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    ListXQueues();

    return 0;
}

int Cli::QueryPolicy()
{
    XPolicyType type;
    XSCHED_ASSERT(client_->QueryPolicy(type));
    std::cout << "Current policy: \n  " << GetPolicyTypeName(type) << std::endl;
    std::cout << "Available policies: " << std::endl;
    for (int i = kPolicyUnknown + 1; i < kPolicyMax; i++) {
        std::cout << "  " << GetPolicyTypeName((XPolicyType)i) << std::endl;
    }
    return 0;
}

int Cli::SetPolicy(const std::string &policy_name)
{
    XPolicyType type = GetPolicyType(policy_name);
    if (type == kPolicyUnknown) XERRO("invalid policy name: %s", policy_name.c_str());
    XSCHED_ASSERT(client_->SetPolicy(type));
    std::cout << "Policy set to " << policy_name << std::endl;
    return 0;
}

int Cli::SetPriority(XQueueHandle handle, Priority prio)
{
    XSCHED_ASSERT(client_->SetPriority(handle, prio));
    std::cout << "Priority of XQueue " << ToHex(handle) << " set to " << prio << std::endl;
    return 0;
}

int Cli::SetProcessPriority(PID pid, Priority prio)
{
    XSCHED_ASSERT(client_->SetProcessPriority(pid, prio));
    return 0;
}

int Cli::SetUtilization(XQueueHandle handle, Utilization util)
{
    XSCHED_ASSERT(client_->SetUtilization(handle, util));
    std::cout << "Utilization of XQueue " << ToHex(handle) << " set to " << util << std::endl;
    return 0;
}

int Cli::SetProcessUtilization(PID pid, Utilization util)
{
    XSCHED_ASSERT(client_->SetProcessUtilization(pid, util));
    std::cout << "Utilization of process " << pid << " set to " << util << std::endl;
    return 0;
}

int Cli::SetTimeslice(Timeslice ts_us)
{
    XSCHED_ASSERT(client_->SetTimeslice(ts_us));
    std::cout << "Timeslice set to " << ts_us << "us\n";
    return 0;
}
