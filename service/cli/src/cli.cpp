#include <set>
#include <sstream>
#include <algorithm>
#include <unordered_map>

#include "cli.h"
#include "convert.h"
#include "xsched/utils/str.h"
#include "xsched/utils/xassert.h"
#include "xsched/protocol/names.h"
#include "xsched/protocol/device.h"
#include "xsched/sched/protocol/hint.h"
#include "xsched/sched/protocol/event.h"
#include "xsched/sched/protocol/names.h"

using namespace tabulate;
using namespace xsched::sched;
using namespace xsched::service;
using namespace xsched::protocol;

Cli::Cli(const std::string& addr, uint16_t port)
    : kAddr(addr), kPort(port)
{
    client_ = std::make_unique<httplib::Client>(kAddr, kPort);
}

Cli::~Cli()
{
    if(client_) client_->stop();
    client_ = nullptr;
}

std::string Cli::ToHex(uint64_t x)
{
    std::stringstream ss;
    ss << "0x" << std::hex << x;
    return ss.str();
}

void Cli::GetResponse(const httplib::Result &res, Json::Value &response)
{
    if (res.error() != httplib::Error::Success || res == nullptr) {
        XERRO("failed to get response, error: %s", httplib::to_string(res.error()).c_str());
    }

    if (res->status != httplib::StatusCode::OK_200) {
        XERRO("failed to list XQueues, response code: %d, message: %s",
              res->status, res->body.c_str());
    }

    if (!json_reader_.parse(res->body, response, false)) {
        XERRO("failed to parse response, message: %s", res->body.c_str());
    }
}

int Cli::ListXQueues()
{
    Json::Value response;
    GetResponse(client_->Get("/xqueues"), response);
    Json::Value xqueues = response["xqueues"];
    Json::Value processes = response["processes"];

    Table table;
    table.add_row({"PID", "DEV", "XQUEUE", "STAT", "SCHED", "LV", "CMD"});
    table.row(0).format().font_style({FontStyle::bold}).font_align(FontAlign::center);

    if (xqueues.empty()) {
        std::cout << table << std::endl;
        return 0;
    }

    std::unordered_map<PID, std::string> pid_to_cmdline;
    for (const auto &process : processes) {
        pid_to_cmdline[process["pid"].asInt()] = process["cmdline"].asString();
    }

    std::vector<XQueueStatus> xqueue_status(xqueues.size());
    size_t idx = 0;
    for (const auto &xqueue : xqueues) {
        JsonToXQueueStatus(xqueue_status[idx++], xqueue);
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
    std::string url = "/config/" + ToHex(handle)
                    + "?level=" + std::to_string((int)level)
                    + "&threshold=" + std::to_string(threshold)
                    + "&batch_size=" + std::to_string(batch_size);
    
    Json::Value response;
    GetResponse(client_->Post(url), response);

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
    Json::Value response;
    GetResponse(client_->Get("/policy"), response);

    PolicyType type = response.isMember("policy")
                    ? (PolicyType)response["policy"].asInt() : kPolicyTypeUnknown;
    std::cout << "Current policy: \n  " << GetPolicyTypeName(type) << std::endl;
    std::cout << "Available policies: " << std::endl;
    for (int i = kPolicyTypeInternalMax + 1; i < kPolicyTypeMax; i++) {
        std::cout << "  " << GetPolicyTypeName((PolicyType)i) << std::endl;
    }
    return 0;
}

int Cli::SetPolicy(const std::string &policy_name)
{
    PolicyType type = GetPolicyType(policy_name);
    if (type == kPolicyTypeUnknown) {
        XERRO("invalid policy name: %s", policy_name.c_str());
    }
    
    Json::Value response;
    std::string url = "/policy?policy=" + std::to_string((int)type);
    GetResponse(client_->Post(url), response);

    std::cout << "Policy set to " << policy_name << std::endl;
    return 0;
}

void Cli::SendHint(const Json::Value &request)
{
    Json::Value response;
    Json::StreamWriterBuilder json_writer;
    GetResponse(client_->Post("/hint",
                              Json::writeString(json_writer, request).c_str(),
                              "application/json"),
                response);
}

int Cli::SetPriority(XQueueHandle handle, Priority prio)
{
    Json::Value request;
    request["hint_type"] = (Json::Int)kHintTypePriority;
    request["handle"] = (Json::UInt64)handle;
    request["priority"] = (Json::Int)prio;

    SendHint(request);

    std::cout << "Priority of XQueue " << ToHex(handle) << " set to " << prio << std::endl;
    return 0;
}

int Cli::SetProcessPriority(PID pid, Priority prio)
{
    Json::Value response;
    GetResponse(client_->Get("/xqueues"), response);
    Json::Value xqueues = response["xqueues"];
    for (const auto &xqueue : xqueues) {
        XQueueStatus status;
        JsonToXQueueStatus(status, xqueue);
        if(status.pid == pid) SetPriority(status.handle, prio);
    }
    return 0;
}

int Cli::SetUtilization(XQueueHandle handle, Utilization util)
{
    Json::Value request;
    request["hint_type"]   = (Json::Int)kHintTypeUtilization;
    request["pid"]         = (Json::Int)0;
    request["handle"]      = (Json::UInt64)handle;
    request["utilization"] = (Json::Int)util;
    SendHint(request);
    std::cout << "Utilization of XQueue " << ToHex(handle) << " set to " << util << std::endl;
    return 0;
}

int Cli::SetProcessUtilization(PID pid, Utilization util)
{
    Json::Value request;
    request["hint_type"]   = (Json::Int)kHintTypeUtilization;
    request["pid"]         = (Json::Int)pid;
    request["handle"]      = (Json::UInt64)0;
    request["utilization"] = (Json::Int)util;
    SendHint(request);
    std::cout << "Utilization of process " << pid << " set to " << util << std::endl;
    return 0;
}

int Cli::SetTimeslice(Timeslice ts_us)
{
    Json::Value request;
    request["hint_type"] = (Json::Int)kHintTypeTimeslice;
    request["timeslice"] = (Json::Int64)ts_us;

    SendHint(request);

    std::cout << "Timeslice set to " << ts_us << "us\n";
    return 0;
}
