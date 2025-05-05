#include <cstring>
#include <algorithm>

#include "xsched/utils/xassert.h"
#include "xsched/sched/protocol/names.h"
#include "xsched/sched/scheduler/local.h"

using namespace std::chrono;
using namespace xsched::sched;

LocalScheduler::LocalScheduler(PolicyType type): Scheduler(kSchedulerLocal), policy_type_(type)
{
    event_queue_ = std::make_unique<std::list<std::shared_ptr<const Event>>>();
    policy_ = CreatePolicy(type);
    policy_->SetSuspendFunc(std::bind(&LocalScheduler::Suspend, this, std::placeholders::_1));
    policy_->SetResumeFunc(std::bind(&LocalScheduler::Resume, this, std::placeholders::_1));
    policy_->SetAddTimerFunc(std::bind(&LocalScheduler::AddTimer, this, std::placeholders::_1));
}

LocalScheduler::~LocalScheduler()
{
    this->Stop();
}

void LocalScheduler::Run()
{
    thread_ = std::make_unique<std::thread>(&LocalScheduler::Worker, this);
}

void LocalScheduler::Stop()
{
    if (thread_) {
        this->RecvEvent(std::make_shared<SchedulerTerminateEvent>());
        thread_->join();
    }

    thread_ = nullptr;
    timers_.clear();
}

void LocalScheduler::RecvEvent(std::shared_ptr<const Event> event)
{
    event_mtx_.lock();
    event_queue_->emplace_back(event);
    event_mtx_.unlock();
    event_cv_.notify_all();
}

void LocalScheduler::SetPolicy(PolicyType type)
{
    std::string old = GetPolicyTypeName(policy_type_);
    this->Stop();
    policy_type_ = type;
    policy_ = CreatePolicy(type);
    policy_->SetSuspendFunc(std::bind(&LocalScheduler::Suspend, this, std::placeholders::_1));
    policy_->SetResumeFunc(std::bind(&LocalScheduler::Resume, this, std::placeholders::_1));
    policy_->SetAddTimerFunc(std::bind(&LocalScheduler::AddTimer, this, std::placeholders::_1));
    for (auto &status : status_.xqueue_status) Resume(status.first);
    this->Run();
    XINFO("policy changed from %s to %s", old.c_str(), GetPolicyTypeName(policy_type_).c_str());
}

void LocalScheduler::Worker()
{
    auto tmp_queue = std::make_unique<std::list<std::shared_ptr<const Event>>>();
    std::unique_lock<std::mutex> lock(event_mtx_);

    while (true) {
        // wait for event or the first timer
        while (event_queue_->empty()) {
            // timers will only be added during policy_->Sched(status_)
            if (timers_.empty()) {
                event_cv_.wait(lock);
                continue;
            }

            auto first_timer = timers_.front();
            auto now = std::chrono::system_clock::now();
            if (now < first_timer) {
                event_cv_.wait_until(lock, first_timer);
                continue;
            }

            while (!timers_.empty()) {
                if (now < timers_.front()) break;
                timers_.pop_front();
            }
            break;
        }
        
        if (!event_queue_->empty()) {
            // swap event_queue_ and tmp_queue
            auto old = std::move(event_queue_);
            event_queue_ = std::move(tmp_queue);
            tmp_queue = std::move(old);
        }
        lock.unlock();

        // process events in tmp_queue
        bool terminate = false;
        while (!tmp_queue->empty()) {
            auto event = tmp_queue->front();
            tmp_queue->pop_front();
            if (UNLIKELY(event->Type() == kEventSchedulerTerminate)) {
                terminate = true; // terminate the worker after processing all events
                continue;
            }
            this->UpdateStatus(event);
        }

        policy_->Sched(status_);    // reschedule
        this->ExecuteOperations();  // find changes and execute
        std::sort(timers_.begin(), timers_.end());
        if (terminate) return;
        lock.lock();
    }
}

void LocalScheduler::ExecuteOperations()
{
    for (auto &status : status_.process_status) {
        if (status.second->running_xqueues.empty() &&
            status.second->suspended_xqueues.empty()) continue;
        Execute(std::make_shared<SchedOperation>(*status.second));
    }
}

void LocalScheduler::CreateXQueueStatus(PID pid, const std::string &cmdline, XQueueHandle handle,
                                        XDevice device, XPreemptLevel level,
                                        int64_t threshold, int64_t batch_size, bool ready,
                                        std::chrono::system_clock::time_point ready_time)
{
    auto status = std::make_unique<XQueueStatus>();
    status->handle = handle;
    status->device = device;
    status->level = level;
    status->pid = pid;
    status->threshold = threshold;
    status->batch_size = batch_size;
    status->ready = ready;
    status->suspended = false;
    status->ready_time = ready_time;
    status_.xqueue_status[handle] = std::move(status);

    // if process status not exist, create one
    auto it = status_.process_status.find(pid);
    if (it == status_.process_status.end()) {
        auto process_status = std::make_unique<ProcessStatus>();
        process_status->info.pid = pid;
        process_status->info.cmdline = cmdline;
        status_.process_status[pid] = std::move(process_status);
        it = status_.process_status.find(pid);
    }
    it->second->running_xqueues.insert(handle);
}

void LocalScheduler::UpdateStatus(std::shared_ptr<const Event> event)
{
    switch (event->Type()) {
    case kEventHint:
    {
        auto e = std::dynamic_pointer_cast<const HintEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        policy_->RecvHint(e->GetHint());
        break;
    }
    case kEventProcessCreate:
    {
        auto e = std::dynamic_pointer_cast<const ProcessCreateEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        PID pid = e->Pid();
        auto it = status_.process_status.find(pid);
        if (it != status_.process_status.end()) {
            it->second->info.cmdline = e->Cmdline();
            break;
        }
        auto process_status = std::make_unique<ProcessStatus>();
        process_status->info.pid = pid;
        process_status->info.cmdline = e->Cmdline();
        status_.process_status[pid] = std::move(process_status);
        break;
    }
    case kEventProcessDestroy:
    {
        auto e = std::dynamic_pointer_cast<const ProcessDestroyEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        PID pid = e->Pid();
        auto pit = status_.process_status.find(pid);
        if (pit == status_.process_status.end()) break;

        for (auto &handle : pit->second->running_xqueues) status_.xqueue_status.erase(handle);
        for (auto &handle : pit->second->suspended_xqueues) status_.xqueue_status.erase(handle);
        status_.process_status.erase(pit);
        break;
    }
    case kEventXQueueCreate:
    {
        auto e = std::dynamic_pointer_cast<const XQueueCreateEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        XINFO("XQueue (0x%lx) from process (%u) created", e->Handle(), e->Pid());
        auto it = status_.xqueue_status.find(e->Handle());
        if (it == status_.xqueue_status.end()) {
            // if xqueue status not exist, create one
            CreateXQueueStatus(e->Pid(), "", e->Handle(), e->Device(),
                               e->Level(), e->Threshold(), e->BatchSize(),
                               false, system_clock::now());
        } else {
            it->second->device = e->Device();
            it->second->level = e->Level();
            it->second->threshold = e->Threshold();
            it->second->batch_size = e->BatchSize();
        }
        break;
    }
    case kEventXQueueDestroy:
    {
        auto e = std::dynamic_pointer_cast<const XQueueDestroyEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        XINFO("XQueue (0x%lx) from process (%u) destroyed", e->Handle(), e->Pid());
        XQueueHandle handle = e->Handle();
        auto qit = status_.xqueue_status.find(handle);
        if (qit == status_.xqueue_status.end()) break;

        PID pid = qit->second->pid;
        XASSERT(e->Pid() == pid, "pid not match");
        auto pit = status_.process_status.find(pid);
        if (pit == status_.process_status.end()) break;

        pit->second->running_xqueues.erase(handle);
        pit->second->suspended_xqueues.erase(handle);
        status_.xqueue_status.erase(qit);
        break;
    }
    case kEventXQueueReady:
    {
        auto e = std::dynamic_pointer_cast<const XQueueReadyEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        auto it = status_.xqueue_status.find(e->Handle());
        if (it == status_.xqueue_status.end()) {
            // if xqueue status not exist, create one
            CreateXQueueStatus(e->Pid(), "", e->Handle(), kDeviceTypeUnknown,
                               kPreemptLevelUnknown, 0, 0, true, e->ReadyTime());
        } else {
            it->second->ready = true;
            it->second->ready_time = e->ReadyTime();
        }
        break;
    }
    case kEventXQueueIdle:
    {
        auto e = std::dynamic_pointer_cast<const XQueueIdleEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        auto it = status_.xqueue_status.find(e->Handle());
        if (it == status_.xqueue_status.end()) {
            // if xqueue status not exist, create one
            CreateXQueueStatus(e->Pid(), "", e->Handle(), kDeviceTypeUnknown,
                               kPreemptLevelUnknown, 0, 0, false, system_clock::now());
        } else {
            it->second->ready = false;
        }
        break;
    }
    case kEventXQueueConfigUpdate:
    {
        auto e = std::dynamic_pointer_cast<const XQueueConfigUpdateEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        auto it = status_.xqueue_status.find(e->Handle());
        if (it == status_.xqueue_status.end()) {
            CreateXQueueStatus(e->Pid(), "", e->Handle(), e->Device(),
                               e->Level(), e->Threshold(), e->BatchSize(),
                               false, system_clock::now());
        } else {
            it->second->device = e->Device();
            it->second->level = e->Level();
            it->second->threshold = e->Threshold();
            it->second->batch_size = e->BatchSize();
        }
        break;
    }
    case kEventXQueueQuery:
    {
        auto e = std::dynamic_pointer_cast<const XQueueQueryEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        StatusQuery *query = e->QueryData();
        auto it = status_.xqueue_status.find(e->Handle());
        if (it == status_.xqueue_status.end()) {
            query->Notify();
            break;
        }
        // XQueue found, add to query
        query->status_.emplace_back(std::make_unique<XQueueStatus>(*it->second));
        if (!query->QueryProcess()) {
            query->Notify();
            break;
        }
        // find process info and add to query
        auto pit = status_.process_status.find(it->second->pid);
        if (pit != status_.process_status.end()) {
            query->processes_.emplace_back(std::make_unique<ProcessInfo>(pit->second->info));
        }
        query->Notify();
        break;
    }
    case kEventXQueueQueryAll:
    {
        auto e = std::dynamic_pointer_cast<const XQueueQueryAllEvent>(event);
        XASSERT(e != nullptr, "event type not match");
        StatusQuery *query = e->QueryData();
        // add all xqueue status to query
        for (auto &status : status_.xqueue_status) {
            query->status_.emplace_back(std::make_unique<XQueueStatus>(*status.second));
        }
        if (!query->QueryProcess()) {
            query->Notify();
            break;
        }
        // find process info and add to query
        for (auto &status : status_.process_status) {
            query->processes_.emplace_back(std::make_unique<ProcessInfo>(status.second->info));
        }
        query->Notify();
        break;
    }
    default:
        break;
    }
}

void LocalScheduler::Suspend(XQueueHandle handle)
{
    auto qit = status_.xqueue_status.find(handle);
    if (qit == status_.xqueue_status.end()) return;

    qit->second->suspended = true;
    PID pid = qit->second->pid;

    auto pit = status_.process_status.find(pid);
    if (pit == status_.process_status.end()) return;
    
    pit->second->running_xqueues.erase(handle);
    pit->second->suspended_xqueues.insert(handle);
}

void LocalScheduler::Resume(XQueueHandle handle)
{
    auto qit = status_.xqueue_status.find(handle);
    if (qit == status_.xqueue_status.end()) return;

    qit->second->suspended = false;
    PID pid = qit->second->pid;

    auto pit = status_.process_status.find(pid);
    if (pit == status_.process_status.end()) return;
    
    pit->second->running_xqueues.insert(handle);
    pit->second->suspended_xqueues.erase(handle);
}

void LocalScheduler::AddTimer(const std::chrono::system_clock::time_point time_point)
{
    timers_.push_back(time_point);
}
