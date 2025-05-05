#include "xsched/sched/protocol/status.h"

using namespace xsched::sched;

void StatusQuery::Wait()
{
    std::unique_lock<std::mutex> lock(mtx_);
    while (!ready_) cv_.wait(lock);
}

void StatusQuery::Notify()
{
    mtx_.lock();
    ready_ = true;
    mtx_.unlock();
    cv_.notify_all();
}

void StatusQuery::Reset()
{
    ready_ = false;
    status_.clear();
    if (kQueryProcess) processes_.clear();
}
