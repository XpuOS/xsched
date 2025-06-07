#include <sstream>

#include "convert.h"

using namespace xsched::sched;

std::string xsched::service::ToHex(uint64_t x)
{
    std::stringstream ss;
    ss << "0x" << std::hex << x;
    return ss.str();
}

void xsched::service::XQueueStatusToJson(Json::Value &json, const XQueueStatus &status)
{
    json["handle"]     = (Json::UInt64)status.handle;
    json["device"]     = (Json::UInt64)status.device;
    json["level"]      = (Json::Int)   status.level;
    json["pid"]        = (Json::Int)   status.pid;
    json["threshold"]  = (Json::Int64) status.threshold;
    json["batch_size"] = (Json::Int64) status.batch_size;
    json["ready"]      = status.ready;
    json["suspended"]  = status.suspended;
}

void xsched::service::JsonToXQueueStatus(XQueueStatus &status, const Json::Value &json)
{
    status.handle     = json.isMember("handle")     ? (XQueueHandle) json["handle"]    .asUInt64() : 0;
    status.device     = json.isMember("device")     ? (XDevice)      json["device"]    .asUInt64() : 0;
    status.level      = json.isMember("level")      ? (XPreemptLevel)json["level"]     .asInt()    : kPreemptLevelUnknown;
    status.pid        = json.isMember("pid")        ? (PID)          json["pid"]       .asInt()    : 0;
    status.threshold  = json.isMember("threshold")  ? (int64_t)      json["threshold"] .asInt64()  : -1;
    status.batch_size = json.isMember("batch_size") ? (int64_t)      json["batch_size"].asInt64()  : -1;
    status.ready      = json.isMember("ready")      ? json["ready"]    .asBool() : false;
    status.suspended  = json.isMember("suspended")  ? json["suspended"].asBool() : false;
}
