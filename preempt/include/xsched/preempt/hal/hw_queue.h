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

    /// @brief Submit HwCommand to the hardware.
    /// @param hal_command The pointer to the HwCommand.
    /// NOTE: This function will be called while holding the submit
    /// worker lock, so it should not do any blocking operations
    /// like synchronizing another queue or command.
    virtual void Launch(std::shared_ptr<HwCommand> hw_cmd) = 0;


    /// @brief Synchronize with the hardware to make sure that all
    ///        HwCommands submitted to HwQueue has been executed
    ///        (or killed if called HwQueue->CancelAllCommands()).
    ///        Do not necessarily mean that these HwCommand has been
    ///        "Completed".
    virtual void Synchronize() = 0;


    /// @brief Cancel all cancelable HwCommands submitted to HwQueue.
    ///        The platform layer should guarantee that the cancel
    ///        will not cause side effects and the re-execution
    ///        of the canceled HwCommands will not cause errors.
    virtual void Deactivate() {}


    /// @brief Resume the execution of the hardware.
    /// @return The idx of the first HwCommand that needs to be
    ///         re-submitted.
    virtual void Reactivate(const CommandLog &log) { UNUSED(log); }

    virtual void Interrupt() {}
    virtual void Restore(const CommandLog &log) { UNUSED(log); }

    virtual XDevice GetDevice() = 0;
    virtual HwQueueHandle GetHandle() = 0;
    virtual bool SupportDynamicLevel() = 0;
    virtual XPreemptLevel GetMaxSupportedLevel() = 0;


    virtual void OnPreemptLevelChange(XPreemptLevel level) { UNUSED(level); }

    /// @brief XQueue create event. Will be called on the launching thread when the XQueue and the
    /// thread are created. Should call platform specific APIs like cuCtxSetCurrent().
    virtual void OnXQueueCreate() {}


    /// @brief HwCommand submit event. Will be called when a HwCommand
    ///        is submitted to XQueue. Can do some pre-processing here.
    /// @param hal_command The HwCommand submitted to the XQueue.
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
