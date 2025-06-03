#pragma once

#include <mutex>
#include <unordered_map>

#include "xsched/types.h"
#include "xsched/utils/common.h"
#include "xsched/preempt/xqueue/xcommand.h"

namespace xsched::preempt
{

class XQueue;

class HwCommand : public XCommand
{
public:
    HwCommand(XCommandProperties props = kCommandPropertyNone);
    virtual ~HwCommand() = default;


    /// @brief Synchronize with the hardware to make sure that
    ///        the HwCommand has been executed (or killed if
    ///        called HwQueue->CancelAllCommands()). Do not
    ///        necessarily mean that the HwCommand has been
    ///        "Completed".
    virtual void Synchronize() = 0;


    /// @brief Check whether the HwCommand currently supports
    ///        synchronization with the hardware. For example,
    ///        CudaEventRecordCommand will return true, because
    ///        we can use cuEventWait() to sync. May return
    ///        false before EnableSynchronization() and true
    ///        after EnableSynchronization().
    /// @return Whether the HwCommand currently supports
    ///         synchronization with the hardware.
    virtual bool Synchronizable() = 0;


    /// @brief Enable synchronization between the HwCommand and
    ///        the hardware. For example, a CudaLaunchKernelCommand
    ///        can be enabled, because we can attach a CudaEvent
    ///        to it and synchronize the CudaEvent.
    /// @return Whether the synchronization has been enabled successfully.
    virtual bool EnableSynchronization() = 0;


    /// @brief Wait until the HwCommand actually become "Completed".
    virtual void Wait() final override;


    /// @brief Will be called before submit to HwQueue. Can do
    ///        synchronization in BeforeHalSubmit(). For example,
    ///        cuStreamWaitEvent can wait until the cuEvent is
    ///        synchronized before HalSubmit.
    /// NOTE: This function will be called without holding the
    ///       submit worker lock.
    virtual void BeforeLaunch() {}


    virtual HwCommandHandle GetHandle() {return (uint64_t)this ^ ((uint64_t)GetProcessId() << 48);}


    /// @brief Get the index of the HwCommand submitted to the XQueue.
    ///        The index starts from 1, available AFTER the HwCommand
    ///        is submitted to the XQueue.
    /// @return The index of the HwCommand.
    int64_t GetIdx() const { return idx_; }
    
    void SetIdx(int64_t idx) { idx_ = idx; }

    /// @brief Get the handle of the XQueue that the HwCommand is submitted to.
    /// @return The handle of the XQueue. 0 if the HwCommand is not submitted to any XQueue.
    XQueueHandle GetXQueueHandle() const { return xqueue_handle_; }


    /// @brief Will be called when the HwCommand is submitted to XQueue.
    /// @param xqueue The pointer to the XQueue.
    void OnSubmit(std::shared_ptr<XQueue> xqueue);

private:
    int64_t idx_ = -1;
    XQueueHandle xqueue_handle_ = 0;
    std::shared_ptr<XQueue> xqueue_ = nullptr;
};

class HwCallbackCommand final : public HwCommand
{
public:
    HwCallbackCommand(LaunchCallback launch, void *data)
        : HwCommand(kCommandPropertyNone), launch_(launch), data_(data) {}
    virtual ~HwCallbackCommand() = default;

    virtual void Synchronize()           override {}
    virtual bool Synchronizable()        override { return false; }
    virtual bool EnableSynchronization() override { return false; }
    XResult Launch(HwQueueHandle hwq) { return launch_(hwq, data_); }

private:
    LaunchCallback launch_ = nullptr;
    void *data_ = nullptr;
};

class HwCommandManager
{
public:
    STATIC_CLASS(HwCommandManager);

    static HwCommandHandle Add(std::shared_ptr<HwCommand> hw_cmd);
    static std::shared_ptr<HwCommand> Del(HwCommandHandle hw_cmd_h);
    static std::shared_ptr<HwCommand> Get(HwCommandHandle hw_cmd_h);

private:
    static std::mutex mtx_;
    static std::unordered_map<HwCommandHandle, std::shared_ptr<HwCommand>> hw_cmds_;
};

} // namespace xsched::preempt
