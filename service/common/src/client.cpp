#include <unordered_map>

#include "client.h"
#include "convert.h"
#include "xsched/utils/log.h"
#include "xsched/sched/protocol/hint.h"

using namespace xsched::sched;
using namespace xsched::service;

Client::Client(const std::string &addr, uint16_t port): kAddr(addr), kPort(port)
{
    http_client_ = std::make_unique<httplib::Client>(kAddr, kPort);
}

Client::~Client()
{
    if(http_client_) http_client_->stop();
    http_client_ = nullptr;
}

XResult Client::SendHint(const Json::Value &request)
{
    Json::Value response;
    std::string request_str = Json::writeString(json_writer_, request);
    return GetResponse(http_client_->Post("/hint", request_str.c_str(), "application/json"),
                       response);
}

XResult Client::GetResponse(const httplib::Result &res, Json::Value &response)
{
    if (res.error() != httplib::Error::Success || res == nullptr) {
        XWARN("failed to get response, error: %s", httplib::to_string(res.error()).c_str());
        return kXSchedErrorBadResponse;
    }

    if (res->status != httplib::StatusCode::OK_200) {
        XWARN("failed to list XQueues, response code: %d, message: %s",
              res->status, res->body.c_str());
        return kXSchedErrorBadResponse;
    }

    if (!json_reader_.parse(res->body, response, false)) {
        XWARN("failed to parse response, message: %s", res->body.c_str());
        return kXSchedErrorBadResponse;
    }

    return kXSchedSuccess;
}

XResult Client::QueryXQueues(std::vector<sched::XQueueStatus> &xqueue_status,
                             std::unordered_map<PID, std::string> &cmdlines)
{
    Json::Value response;
    GetResponse(http_client_->Get("/xqueues"), response);
    Json::Value xqueues = response["xqueues"];
    Json::Value processes = response["processes"];
    if (xqueues.empty()) return kXSchedSuccess;

    size_t idx = 0;
    xqueue_status.resize(xqueues.size());
    for (const auto &xqueue : xqueues) {
        JsonToXQueueStatus(xqueue_status[idx++], xqueue);
    }
    for (const auto &process : processes) {
        cmdlines[process["pid"].asInt()] = process["cmdline"].asString();
    }

    return kXSchedSuccess;
}

XResult Client::SetXQueueConfig(XQueueHandle handle, XPreemptLevel level,
                                int64_t threshold, int64_t batch_size)
{
    std::string url = "/config/" + ToHex(handle)
                    + "?level=" + std::to_string((int)level)
                    + "&threshold=" + std::to_string(threshold)
                    + "&batch_size=" + std::to_string(batch_size);
    
    Json::Value response;
    return GetResponse(http_client_->Post(url), response);
}

XResult Client::QueryPolicy(XPolicyType &policy)
{
    Json::Value resp;
    XResult res = GetResponse(http_client_->Get("/policy"), resp);
    if (res != kXSchedSuccess) return res;
    policy = resp.isMember("policy") ? (XPolicyType)resp["policy"].asInt() : kPolicyUnknown;
    return kXSchedSuccess;
}

XResult Client::SetPolicy(XPolicyType policy)
{
    Json::Value response;
    std::string url = "/policy?policy=" + std::to_string((int)policy);
    return GetResponse(http_client_->Post(url), response);
}

XResult Client::SetPriority(XQueueHandle handle, Priority prio)
{
    Json::Value request;
    request["hint_type"] = (Json::Int)kHintTypePriority;
    request["handle"] = (Json::UInt64)handle;
    request["priority"] = (Json::Int)prio;
    return SendHint(request);
}

XResult Client::SetProcessPriority(PID pid, Priority prio)
{
    Json::Value response;
    XResult res = GetResponse(http_client_->Get("/xqueues"), response);
    if (res != kXSchedSuccess) return res;

    // set priority for all XQueues of the process
    Json::Value xqueues = response["xqueues"];
    for (const auto &xqueue : xqueues) {
        XQueueStatus status;
        JsonToXQueueStatus(status, xqueue);
        if(status.pid == pid) {
            XResult set_res = SetPriority(status.handle, prio);
            if (set_res != kXSchedSuccess) return set_res;
        }
    }

    return kXSchedSuccess;
}

XResult Client::SetUtilization(XQueueHandle handle, Utilization util)
{
    Json::Value request;
    request["hint_type"]   = (Json::Int)kHintTypeUtilization;
    request["pid"]         = (Json::Int)0;
    request["handle"]      = (Json::UInt64)handle;
    request["utilization"] = (Json::Int)util;
    return SendHint(request);
}

XResult Client::SetProcessUtilization(PID pid, Utilization util)
{
    Json::Value request;
    request["hint_type"]   = (Json::Int)kHintTypeUtilization;
    request["pid"]         = (Json::Int)pid;
    request["handle"]      = (Json::UInt64)0;
    request["utilization"] = (Json::Int)util;
    return SendHint(request);
}

XResult Client::SetTimeslice(Timeslice ts_us)
{
    Json::Value request;
    request["hint_type"] = (Json::Int)kHintTypeTimeslice;
    request["timeslice"] = (Json::Int64)ts_us;
    return SendHint(request);
}

XResult Client::SetActiveWindow(PID pid, uint64_t display)
{
    Json::Value request;
    request["hint_type"] = (Json::Int)kHintTypeWindowActive;
    request["pid"]       = (Json::Int)pid;
    request["display"]   = (Json::UInt64)display;
    return SendHint(request);
}
