#pragma once

#include <mutex>
#include <memory>
#include <thread>
#include <string>
#include <unordered_map>

#define CPPHTTPLIB_THREAD_POOL_COUNT 2
#include <httplib.h>
#include <json/json.h>
#include <libipc/ipc.h>

#include "xsched/types.h"
#include "xsched/utils/waitpid.h"
#include "xsched/sched/scheduler/local.h"
#include "xsched/sched/protocol/status.h"

namespace xsched::service
{

class Server
{
public:
    Server(const std::string &policy_name, const std::string &port);
    ~Server();

    void Run();
    void Stop();

private:
    void RecvWorker();
    void CleanUpProcess(PID pid);
    void ProcessTerminate(PID pid);
    void SendHint(std::shared_ptr<const sched::Hint> hint);
    void Execute(std::shared_ptr<const sched::Operation> operation);

    XQueueHandle GetXQueueHandle(const Json::Value &request);
    bool GetXQueueStatus(XQueueHandle handle, sched::XQueueStatus &status);

    void GetXQueue(const httplib::Request &req, httplib::Response &res);
    void GetXQueues(const httplib::Request &req, httplib::Response &res);
    void PostXQueueConfig(const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &reader);
    void GetSchedulerPolicy(const httplib::Request &req, httplib::Response &res);
    void PostSchedulerPolicy(const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &reader);
    void PostHint(const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &reader);

    std::mutex chan_mtx_;
    std::unique_ptr<ipc::channel> recv_chan_ = nullptr;
    std::unique_ptr<ipc::channel> self_chan_ = nullptr;
    std::unordered_map<PID, std::shared_ptr<ipc::channel>> client_chans_;
    std::unique_ptr<utils::PidWaiter> pid_waiter_ = nullptr;
    std::unique_ptr<sched::LocalScheduler> scheduler_ = nullptr;
    
    uint16_t port_ = 0;
    Json::Reader json_reader_;
    Json::StreamWriterBuilder json_writer_;
    httplib::Server http_server_;
    std::unique_ptr<std::thread> http_thread_ = nullptr;
};

} // namespace xsched::service
