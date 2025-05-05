#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <functional>

namespace xsched::utils
{

class LoopRunner
{
public:
    LoopRunner() = default;
    ~LoopRunner();

    void Start(std::function<void()> loop_body);
    void Stop();
    bool Running() const { return running_.load(); }

private:
    std::atomic_bool running_{false};
    std::unique_ptr<std::thread> thread_ = nullptr;

    void Loop(std::function<void()> loop_body);
};

} // namespace xsched::utils
