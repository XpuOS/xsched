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
    client_chn_key_ = (ipc::ChannelKey)GetProcessId();
    ipc::ChannelKey server_chn_key = (ipc::ChannelKey)XSCHED_SERVER_CHANNEL_KEY;
    recv_chan_ = std::make_unique<ipc::Node>(client_chn_key_, ipc::NodeType::kReceiver);
    send_chan_ = std::make_unique<ipc::Node>(server_chn_key, ipc::NodeType::kSender);
    thread_ = std::make_unique<std::thread>(&GlobalScheduler::Worker, this);
}

void GlobalScheduler::Stop()
{
    if (thread_) {
        auto op = std::make_unique<TerminateOperation>();
        auto self_chan = std::make_unique<ipc::Node>(client_chn_key_, ipc::NodeType::kSender);
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
            XDEBG("recv_chan_ receive failed, exiting Worker thread");
            return;
        }
        auto op = Operation::CopyConstructor(data->Data());
        if (UNLIKELY(op->Type() == kOperationTerminate)) return;
        Execute(op);
    }
}
