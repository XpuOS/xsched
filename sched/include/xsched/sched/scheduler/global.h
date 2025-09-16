#pragma once

#include <memory>
#include <thread>
#include <ipc/ipc.h>

#include "xsched/sched/scheduler/scheduler.h"

namespace xsched::sched
{

class GlobalScheduler : public Scheduler
{
public:
    GlobalScheduler();
    virtual ~GlobalScheduler();

    virtual void Run() override;
    virtual void Stop() override;
    virtual void RecvEvent(std::shared_ptr<const Event> event) override;

private:
    void Worker();

    std::string client_chan_name_;
    std::unique_ptr<std::thread> thread_;
    std::unique_ptr<ipc::Node> recv_chan_;
    std::unique_ptr<ipc::Node> send_chan_;
};

} // namespace xsched::sched
