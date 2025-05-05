#pragma once

#include <list>
#include <mutex>
#include <memory>
#include <functional>
#include <unordered_map>

#include "xsched/utils.h"
#include "xsched/xqueue.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::preempt
{

class XQueue;
using CommandLog = std::list<std::shared_ptr<HwCommand>>;

class HwQueue
{
public:
    HwQueue() = default;
    virtual ~HwQueue() = default;

    /// @brief Level-1 (kPreemptLevelBlock) interface. Launch a HwCommand on this HwQueue.
    /// @param hw_cmd The pointer to the HwCommand.
    /// NOTE: This function will be called while holding the launch worker lock, so it should
    /// NOT do any blocking operations like synchronizing another HwQueue or HwCommand.
    virtual void Launch(std::shared_ptr<HwCommand> hw_cmd) = 0;

    /// @brief Level-1 (kPreemptLevelBlock) interface. Synchronize with the hardware to wait until
    /// ALL HwCommands launched on this HwQueue are finished. Can ALSO return after the HwCommands
    /// are flushed when deactivating the HwQueue, or killed when interrupting the HwQueue.
    virtual void Synchronize() = 0;

    /// @brief Level-2 (kPreemptLevelDeactivate) interface. Deactivate the HwQueue to prevent new
    /// in-flight HwCommands in this HwQueue from being fetched and executed by the hardware. Can
    /// be implemented by stalling HwCommand dequeuing, or flushing away the in-flight HwCommands.
    virtual void Deactivate() {}

    /// @brief Level-2 (kPreemptLevelDeactivate) interface. Resume a deactivated HwQueue.
    /// @param log In-flight but not completed HwCommands that have been launched on this HwQueue.
    virtual void Reactivate(const CommandLog &log) { UNUSED(log); }

    /// @brief Level-3 (kPreemptLevelInterrupt) interface. Interrupt a HwQueue to immediately stop
    /// the currently running HwCommand. Can be implemented by sending interrupt signals to the
    /// hardware to kill the running HwCommand or trigger context switch.
    virtual void Interrupt() {}

    /// @brief Level-3 (kPreemptLevelInterrupt) interface. Restore an interrupted HwQueue.
    /// @param log In-flight but not completed HwCommands that have been launched on this HwQueue.
    virtual void Restore(const CommandLog &log) { UNUSED(log); }

    virtual XDevice GetDevice() = 0;
    virtual HwQueueHandle GetHandle() = 0;
    virtual bool SupportDynamicLevel() = 0;
    virtual XPreemptLevel GetMaxSupportedLevel() = 0;

    /// @brief A callback function that will be called when the preemption level changes.
    /// @param level The new preemption level of the HwQueue.
    virtual void OnPreemptLevelChange(XPreemptLevel level) { UNUSED(level); }

    /// @brief A callback function that will be called on the launching thread when the XQueue and
    /// the thread are created. Can call platform specific APIs like cuCtxSetCurrent().
    virtual void OnXQueueCreate() {}

    /// @brief A callback function that will be called when a HwCommand is submitted to XQueue.
    /// Can do some pre-processing here, e.g., instrumenting a CUDA kernel function.
    /// @param hw_cmd The HwCommand submitted to the XQueue.
    virtual void OnHwCommandSubmit(std::shared_ptr<HwCommand> hw_cmd) { UNUSED(hw_cmd); }

    std::shared_ptr<XQueue> GetXQueue() { return xq_; }
    void SetXQueue(std::shared_ptr<XQueue> xq) { xq_ = xq; }

private:
    std::shared_ptr<XQueue> xq_ = nullptr;
};

class HwQueueManager
{
public:
    STATIC_CLASS(HwQueueManager);

    static XResult Add(HwQueueHandle hwq_h, std::function<std::shared_ptr<HwQueue>()> create);
    static XResult Del(HwQueueHandle hwq_h);
    static std::shared_ptr<HwQueue> Get(HwQueueHandle hwq_h);
    static std::shared_ptr<XQueue> GetXQueue(HwQueueHandle hwq_h);

private:
    static std::mutex mtx_;
    static std::unordered_map<HwQueueHandle, std::shared_ptr<HwQueue>> hwqs_;
};

} // namespace xsched::preempt
