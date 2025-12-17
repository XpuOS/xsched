#include <mutex>

#include "server.h"
#include "convert.h"
#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/utils/common.h"
#include "xsched/protocol/def.h"
#include "xsched/protocol/names.h"
#include "xsched/sched/protocol/operation.h"

using namespace xsched::utils;
using namespace xsched::sched;
using namespace xsched::service;
using namespace xsched::protocol;

Server::Server(const std::string &policy_name, const std::string &port)
{
    port_ = std::stoi(port);

    XPolicyType type = GetPolicyType(policy_name);
    if (type == kPolicyUnknown) XERRO("invalid policy name %s", policy_name.c_str());
    XASSERT(type > kPolicyUnknown && type < kPolicyMax, "server policy must be a customized policy");

    scheduler_ = std::make_unique<LocalScheduler>(type);
    XINFO("scheduler created with policy %s", policy_name.c_str());

    http_server_.Get("/xqueue/:handle", std::bind(&Server::GetXQueue, this,
        std::placeholders::_1, std::placeholders::_2));
    http_server_.Get("/xqueues", std::bind(&Server::GetXQueues, this,
        std::placeholders::_1, std::placeholders::_2));
    http_server_.Post("/config/:handle", std::bind(&Server::PostXQueueConfig, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    http_server_.Get("/policy", std::bind(&Server::GetSchedulerPolicy, this,
        std::placeholders::_1, std::placeholders::_2));
    http_server_.Post("/policy", std::bind(&Server::PostSchedulerPolicy, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    http_server_.Post("/hint", std::bind(&Server::PostHint, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

Server::~Server()
{
    Stop();
}

void Server::Run()
{
    std::string server_name(XSCHED_SERVER_CHANNEL_NAME);
    recv_chan_ = std::make_unique<ipc::Node>(server_name.c_str(), ipc::NodeType::kReceiver);
    self_chan_ = std::make_unique<ipc::Node>(server_name.c_str(), ipc::NodeType::kSender);

    scheduler_->SetExecutor(std::bind(&Server::Execute, this, std::placeholders::_1));
    scheduler_->Run();

    pid_waiter_ = PidWaiter::Create(
                  std::bind(&Server::ProcessTerminate, this, std::placeholders::_1));
    pid_waiter_->Start();

    http_thread_ = std::make_unique<std::thread>([this]() {
        XINFO("HTTP server listening on port %d", port_);
        http_server_.listen("0.0.0.0", port_);
    });

    this->RecvWorker();
}

void Server::Stop()
{
    if (http_thread_) {
        http_server_.stop();
        http_thread_->join();
        http_thread_ = nullptr;
    }

    if (pid_waiter_) {
        pid_waiter_->Stop();
        pid_waiter_ = nullptr;
    }

    if (recv_chan_) {
        // recv_chan_->Receive() will fail and return nullptr
        // FIXME: possible error when calling Stop() twice,
        // recv_chan_ could be set to nullptr here by RecvWorker()
        recv_chan_->Remove();
    }
}

void Server::RecvWorker()
{
    XINFO("events receiver started");

    while (true) {
        std::shared_ptr<const Event> e = nullptr;
        auto data = recv_chan_->Receive();
        if(UNLIKELY(data == nullptr)) {
            XDEBG("channel %s receive fail, exiting RecvWorker thread", recv_chan_->getName().c_str());
            e = std::make_shared<SchedulerTerminateEvent>();
        } else {
            e = Event::CopyConstructor(data->Data());
        }

        switch (e->Type())
        {
        case kEventSchedulerTerminate:
            scheduler_->Stop();
            recv_chan_->Remove();
            self_chan_->Remove();
            for (auto &it : client_chans_) it.second->Remove();

            scheduler_ = nullptr;
            recv_chan_ = nullptr;
            self_chan_ = nullptr;
            client_chans_.clear();
            return;
        case kEventProcessCreate:
        {
            PID client_pid = e->Pid();
            std::string client_name = std::string(XSCHED_CLIENT_CHANNEL_PREFIX)
                                    + std::to_string(client_pid);
            auto client_chan = std::make_shared<ipc::Node>(client_name.c_str(), ipc::NodeType::kSender);
            XINFO("client process " FMT_PID " connected", client_pid);

            chan_mtx_.lock();
            if(client_chans_.count(client_pid) == 0) {
                pid_waiter_->AddWait(client_pid);
            }
            client_chans_[client_pid] = client_chan;
            chan_mtx_.unlock();
            scheduler_->RecvEvent(e);
            break;
        }
        case kEventProcessDestroy:
        {
            scheduler_->RecvEvent(e);
            this->CleanUpProcess(e->Pid());
            break;
        }
        default:
            scheduler_->RecvEvent(e);
            break;
        }
    }
}

void Server::CleanUpProcess(PID pid)
{
    {
        std::lock_guard<std::mutex> lock(chan_mtx_);
        auto it = client_chans_.find(pid);
        if (it == client_chans_.end()) return;
        // Could cause Execute() cannot find the channel,
        // because scheduler_->RecvEvent(e) is asynchronous.
        client_chans_.erase(it);
    }
    XINFO("client process " FMT_PID " closed", pid);
}

void Server::ProcessTerminate(PID pid)
{
    auto e = std::make_shared<ProcessDestroyEvent>(pid);
    scheduler_->RecvEvent(e);
    CleanUpProcess(pid);
}

void Server::SendHint(std::shared_ptr<const sched::Hint> hint)
{
    auto e = std::make_shared<HintEvent>(hint);
    scheduler_->RecvEvent(e);
}

void Server::Execute(std::shared_ptr<const Operation> operation)
{
    PID client_pid = operation->Pid();

    chan_mtx_.lock();
    auto it = client_chans_.find(client_pid);

    if (it == client_chans_.end()) {
        chan_mtx_.unlock();
        // It is possible that the server has received ProcessDestroyEvent in RecvWorker(),
        // and the channel has been closed before the scheduler processes the event.
        XDEBG("cannot find client channel for client process " FMT_PID, client_pid);
        return;
    }

    std::shared_ptr<ipc::Node> client_chan = it->second;
    chan_mtx_.unlock();

    XASSERT(client_chan->Send(operation->Data(), operation->Size()),
            "cannot send operation to client process " FMT_PID, client_pid);
}

XQueueHandle Server::GetXQueueHandle(const Json::Value &request)
{
    if (!request.isMember("handle")) return 0;
    return (XQueueHandle)request["handle"].asUInt64();
}

bool Server::GetXQueueStatus(XQueueHandle handle, XQueueStatus &status)
{
    static std::mutex query_mtx;
    std::lock_guard<std::mutex> lock(query_mtx);

    static StatusQuery query(false); // only query XQueue status
    query.Reset();
    query.status_.reserve(128);

    auto e = std::make_unique<XQueueQueryEvent>(handle, &query);
    XASSERT(self_chan_->Send(e->Data(), e->Size()), "cannot send XQueue query event");
    query.Wait();

    if (query.status_.empty()) return false;
    status = *query.status_[0];
    return true;
}

void Server::GetXQueue(const httplib::Request &req, httplib::Response &res)
{
    size_t pos = 0;
    std::string handle_str = req.path_params.at("handle");
    if (handle_str.empty()) {
        res.status = httplib::StatusCode::BadRequest_400;
        return;
    }
    XQueueHandle handle = std::stoull(handle_str, &pos, 16);
    if (pos != handle_str.length() || handle == 0) {
        res.status = httplib::StatusCode::BadRequest_400;
        return;
    }

    static std::mutex query_mtx;
    std::lock_guard<std::mutex> lock(query_mtx);

    static StatusQuery query(true); // also query process info
    query.Reset();
    query.status_.reserve(128);
    query.processes_.reserve(128);

    auto e = std::make_unique<XQueueQueryEvent>(handle, &query);
    XASSERT(self_chan_->Send(e->Data(), e->Size()), "cannot send XQueue query event");
    query.Wait();

    if (query.status_.empty() || query.processes_.empty()) {
        res.status = httplib::StatusCode::NotFound_404;
        return;
    }
    Json::Value xqueue(Json::objectValue);
    XQueueStatusToJson(xqueue, *query.status_[0]);
    Json::Value process(Json::objectValue);
    process["pid"] = (Json::Int)query.processes_[0]->pid;
    process["cmdline"] = query.processes_[0]->cmdline;

    Json::Value response(Json::objectValue);
    response["xqueue"] = xqueue;
    response["process"] = process;
    res.set_content(Json::writeString(json_writer_, response).c_str(), "application/json");
}

void Server::GetXQueues(const httplib::Request &, httplib::Response &res)
{
    static std::mutex query_mtx;
    std::lock_guard<std::mutex> lock(query_mtx);

    static StatusQuery query(true);
    query.Reset();
    query.status_.reserve(256);
    query.processes_.reserve(256);
    
    auto e = std::make_unique<XQueueQueryAllEvent>(&query);
    XASSERT(self_chan_->Send(e->Data(), e->Size()), "cannot send XQueue query all event");
    query.Wait();

    Json::Value xqueues(Json::arrayValue);
    for (const auto &xq : query.status_) {
        Json::Value xqueue;
        XQueueStatusToJson(xqueue, *xq);
        xqueues.append(xqueue);
    }

    Json::Value processes(Json::arrayValue);
    for (const auto &p : query.processes_) {
        Json::Value process;
        process["pid"] = (Json::Int)p->pid;
        process["cmdline"] = p->cmdline;
        processes.append(process);
    }

    Json::Value response(Json::objectValue);
    response["xqueues"] = xqueues;
    response["processes"] = processes;
    res.set_content(Json::writeString(json_writer_, response).c_str(), "application/json");
}

void Server::PostXQueueConfig(const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &)
{
    size_t pos = 0;
    std::string handle_str = req.path_params.at("handle");
    if (handle_str.empty()) {
        res.status = httplib::StatusCode::BadRequest_400;
        return;
    }
    XQueueHandle handle = std::stoull(handle_str, &pos, 16);
    if (pos != handle_str.length() || handle == 0) {
        res.status = httplib::StatusCode::BadRequest_400;
        return;
    }

    XPreemptLevel level = kPreemptLevelUnknown;
    if (req.has_param("level")) {
        size_t level_pos = 0;
        std::string level_str = req.get_param_value("level");
        level = (XPreemptLevel)std::stoi(level_str, &level_pos, 10);
        if (level_pos != level_str.length() || level <= kPreemptLevelUnknown || level >= kPreemptLevelMax) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"invalid level\"}", "application/json");
            return;
        }
    }

    int64_t threshold = -1;
    if (req.has_param("threshold")) {
        size_t threshold_pos = 0;
        std::string threshold_str = req.get_param_value("threshold");
        threshold = std::stoll(threshold_str, &threshold_pos, 10);
        if (threshold_pos != threshold_str.length() || threshold <= 0) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"invalid command threshold\"}", "application/json");
            return;
        }
    }

    int64_t batch_size = -1;
    if (req.has_param("batch_size")) {
        size_t sync_pos = 0;
        std::string sync_str = req.get_param_value("batch_size");
        batch_size = std::stoll(sync_str, &sync_pos, 10);
        if (sync_pos != sync_str.length() || batch_size <= 0) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"invalid command batch size\"}", "application/json");
            return;
        }
    }

    XQueueStatus status;
    if (!GetXQueueStatus(handle, status)) {
        res.status = httplib::StatusCode::NotFound_404;
        res.set_content("{\"error\": \"XQueue not found\"}", "application/json");
        return;
    }

    if (level != kPreemptLevelUnknown) status.level = level;
    if (threshold > 0) status.threshold = threshold;
    if (batch_size > 0) status.batch_size = batch_size;

    if (status.batch_size >= status.threshold) {
        res.status = httplib::StatusCode::BadRequest_400;
        res.set_content("{\"error\": \"batch size must be less than threshold\"}", "application/json");
        return;
    }

    if (level == kPreemptLevelUnknown && threshold == -1 && batch_size == -1) {
        // no change
        res.set_content("{\"warning\": \"no change\"}", "application/json");
        return;
    }

    Execute(std::make_shared<ConfigOperation>(status.pid, handle, level, threshold, batch_size));
    res.set_content("{\"info\": \"success\"}", "application/json");
}

void Server::GetSchedulerPolicy(const httplib::Request &, httplib::Response &res)
{
    XPolicyType type = scheduler_->GetPolicy();
    Json::Value response(Json::objectValue);
    response["policy"] = (Json::Int)type;
    res.set_content(Json::writeString(json_writer_, response).c_str(), "application/json");
}

void Server::PostSchedulerPolicy(const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &)
{
    if (!req.has_param("policy")) {
        res.status = httplib::StatusCode::BadRequest_400;
        return;
    }

    size_t pos = 0;
    std::string type_str = req.get_param_value("policy");
    if (type_str.empty()) {
        res.status = httplib::StatusCode::BadRequest_400;
        return;
    }

    XPolicyType type = (XPolicyType)std::stoi(type_str, &pos, 10);
    if (pos != type_str.length() || type <= kPolicyUnknown || type >= kPolicyMax) {
        res.status = httplib::StatusCode::BadRequest_400;
        return;
    }

    scheduler_->SetPolicy(type);
    res.set_content("{\"info\": \"success\"}", "application/json");
}

void Server::PostHint(const httplib::Request &, httplib::Response &res, const httplib::ContentReader &reader)
{
    std::string body;
    reader([&](const char *data, size_t data_length) {
        body.append(data, data_length);
        return true;
    });

    Json::Value request;
    if (!json_reader_.parse(body.c_str(), request, false)) {
        res.status = httplib::StatusCode::BadRequest_400;
        return;
    }

    if (!request.isMember("hint_type")) {
        res.status = httplib::StatusCode::BadRequest_400;
        res.set_content("{\"error\": \"missing hint type\"}", "application/json");
        return;
    }

    HintType hint_type = (HintType)request["hint_type"].asInt();
    if (hint_type <= kHintTypeUnknown || hint_type >= kHintTypeMax) {
        res.status = httplib::StatusCode::BadRequest_400;
        res.set_content("{\"error\": \"invalid hint type\"}", "application/json");
        return;
    }

    switch (hint_type) {
    case kHintTypePriority:
    {
        XQueueHandle handle = GetXQueueHandle(request);
        if (handle == 0) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"invalid handle\"}", "application/json");
            return;
        }

        if (!request.isMember("priority")) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"missing priority\"}", "application/json");
            return;
        }

        Priority prio = (Priority)request["priority"].asInt();
        if (prio < PRIORITY_MIN || prio > PRIORITY_MAX) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"invalid priority\"}", "application/json");
            return;
        }

        SendHint(std::make_shared<PriorityHint>(handle, prio));
        res.set_content("{\"info\": \"success\"}", "application/json");
        return;
    }
    case kHintTypeUtilization:
    {
        XQueueHandle handle = GetXQueueHandle(request);
        PID pid = request.isMember("pid") ? (PID)request["pid"].asInt() : 0;

        if (handle == 0 && pid == 0) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"handle or pid is required\"}", "application/json");
            return;
        }

        if (!request.isMember("utilization")) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"missing utilization\"}", "application/json");
            return;
        }

        Utilization util = (Utilization)request["utilization"].asInt();
        if (util < UTILIZATION_MIN || util > UTILIZATION_MAX) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"invalid utilization\"}", "application/json");
            return;
        }

        SendHint(std::make_shared<UtilizationHint>(pid, handle, util));
        res.set_content("{\"info\": \"success\"}", "application/json");
        return;
    }
    case kHintTypeTimeslice:
    {
        if (!request.isMember("timeslice")) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"missing timeslice\"}", "application/json");
            return;
        }

        uint64_t timeslice = request["timeslice"].asUInt64();
        if (timeslice <= 0) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"invalid timeslice\"}", "application/json");
            return;
        }

        SendHint(std::make_shared<TimesliceHint>(timeslice));
        res.set_content("{\"info\": \"success\"}", "application/json");
        return;
    }
    case kHintTypeWindowActive:
    {
        if (!request.isMember("pid")) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"missing pid\"}", "application/json");
            return;
        }

        if (!request.isMember("display")) {
            res.status = httplib::StatusCode::BadRequest_400;
            res.set_content("{\"error\": \"missing display\"}", "application/json");
            return;
        }

        PID pid = (PID)request["pid"].asInt();
        uint64_t display = request["display"].asUInt64();
        SendHint(std::make_shared<WindowActiveHint>(pid, display));
        res.set_content("{\"info\": \"success\"}", "application/json");
        return;
    }
    default:
        res.status = httplib::StatusCode::BadRequest_400;
        res.set_content("{\"error\": \"invalid hint type\"}", "application/json");
        return;
    }

    res.status = httplib::StatusCode::InternalServerError_500;
    res.set_content("{\"error\": \"internal server error\"}", "application/json");
}
