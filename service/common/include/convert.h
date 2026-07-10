#pragma once

#include <json/json.h>
#include "xsched/sched/protocol/status.h"

namespace xsched::service
{

void XQueueStatusToJson(Json::Value &json, const sched::XQueueStatus &status);
void JsonToXQueueStatus(sched::XQueueStatus &status, const Json::Value &json);

} // namespace xsched::service
