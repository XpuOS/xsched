#pragma once

#include <json/json.h>
#include "xsched/sched/protocol/status.h"

namespace xsched::service
{

std::string ToHex(uint64_t x);
void XQueueStatusToJson(Json::Value &json, const sched::XQueueStatus &status);
void JsonToXQueueStatus(sched::XQueueStatus &status, const Json::Value &json);

} // namespace xsched::service
