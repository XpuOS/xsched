#include "xsched/protocol/def.h"
#include "xsched/utils/xassert.h"
#include "xsched/sched/scheduler/global.h"

using namespace xsched::sched;

GlobalScheduler::GlobalScheduler(): Scheduler(kSchedulerGlobal)
{

}

GlobalScheduler::~GlobalScheduler()
{
    this->Stop();
}

void GlobalScheduler::Run()
{
    std::string server_name(XSCHED_SERVER_CHANNEL_NAME);
    client_chan_name_ = std::string(XSCHED_CLIENT_CHANNEL_PREFIX)+ std::to_string(GetProcessId());
    recv_chan_ = std::make_unique<ipc::Node>(client_chan_name_.c_str(), ipc::NodeType::kReceiver);
    send_chan_ = std::make_unique<ipc::Node>(server_name.c_str(), ipc::NodeType::kSender);
    thread_ = std::make_unique<std::thread>(&GlobalScheduler::Worker, this);
}

void GlobalScheduler::Stop()
{
    if (thread_) {
        auto op = std::make_unique<TerminateOperation>();
        auto self_chan = std::make_unique<ipc::Node>(client_chan_name_.c_str(), ipc::NodeType::kSender);
        XASSERT(self_chan->Send(op->Data(), op->Size()),
                "cannot send TerminateOperation to worker thread");
        self_chan->Remove();
        thread_->join();
        thread_ = nullptr;
    }

    if (recv_chan_) {
        recv_chan_->Remove();
        recv_chan_ = nullptr;
    }

    if (send_chan_) {
        send_chan_->Remove();
        send_chan_ = nullptr;
    }
}

void GlobalScheduler::RecvEvent(std::shared_ptr<const Event> event)
{
    bool sent = send_chan_->Send(event->Data(), event->Size());
    if (LIKELY(sent)) return;
    this->Stop();
    XASSERT(false, "failed to send event to server");
}

void GlobalScheduler::Worker()
{
    while (true) {
        auto data = recv_chan_->Receive();
        if (data == nullptr) {
            XDEBG("channel %s receive fail, exiting Worker thread", recv_chan_->getName().c_str());
            return;
        }
        auto op = Operation::CopyConstructor(data->Data());
        if (UNLIKELY(op->Type() == kOperationTerminate)) return;
        Execute(op);
    }
}
