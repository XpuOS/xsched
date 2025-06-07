#include <algorithm>
#include <unordered_map>
#include <tabulate/table.hpp>

#include "cli.h"
#include "convert.h"
#include "xsched_assert.h"
#include "xsched/utils/log.h"
#include "xsched/protocol/names.h"
#include "xsched/protocol/device.h"

using namespace tabulate;
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

    Table table;
    table.add_row({"PID", "DEV", "XQUEUE", "STAT", "SCHED", "LV", "CMD"});
    table.row(0).format().font_style({FontStyle::bold}).font_align(FontAlign::center);

    if (xqueue_status.empty()) {
        std::cout << table << std::endl;
        return 0;
    }

    std::sort(xqueue_status.begin(), xqueue_status.end(),
              [](const XQueueStatus &a, const XQueueStatus &b) { return a.pid < b.pid; });

    size_t row = 1;
    for (const auto &status : xqueue_status) {
        table.add_row({
            std::to_string(status.pid),
            GetDeviceTypeName(GetDeviceType(status.device)) + "(" + ToHex(status.device) + ")",
            ToHex(status.handle),
            status.ready     ? "RDY" : "BLK",
            status.suspended ? "SUS" : "RUN",
            std::to_string((int)status.level),
            pid_to_cmdline[status.pid].substr(0, 60),
        });

        if (status.ready) {
            table[row][3].format().font_color(Color::cyan);
        } else {
            table[row][3].format().font_color(Color::yellow);
        }

        if (status.suspended) {
            table[row][4].format().font_color(Color::red);
        } else {
            table[row][4].format().font_color(Color::green);
        }

        for (size_t i = 0; i < 6; i++) {
            table[row][i].format().font_align(FontAlign::center);
        }
        // Set command column to left align
        table[row][6].format().font_align(FontAlign::left);

        row++;
    }

    std::cout << table << std::endl;
    return 0;
}

int Cli::Top(uint64_t interval_ms)
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
