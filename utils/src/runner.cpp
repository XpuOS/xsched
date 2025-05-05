#include "xsched/utils/runner.h"
#include "xsched/utils/xassert.h"

using namespace xsched::utils;

LoopRunner::~LoopRunner()
{
    if (running_.load()) Stop();
}

void LoopRunner::Start(std::function<void()> loop_body)
{
    XASSERT(!running_.load(), "LoopRunner is already running.");
    
    running_.store(true);
    thread_ = std::make_unique<std::thread>(&LoopRunner::Loop, this, loop_body);
}

void LoopRunner::Stop()
{
    XASSERT(running_.load(), "LoopRunner has already stopped.");

    running_ = false;
    thread_->join();
    thread_ = nullptr;
}

void LoopRunner::Loop(std::function<void()> loop_body)
{
    while (running_.load()) { loop_body(); }
}
