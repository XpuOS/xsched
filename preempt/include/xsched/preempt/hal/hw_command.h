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

    /// @brief Synchronize with the hardware to wait until the HwCommand is finished. Can ALSO
    /// return after the HwCommand is flushed when deactivating the HwQueue, or killed when
    /// interrupting the HwQueue. Must be called after the synchronization is enabled and this
    /// HwCommand has been launched on a HwQueue.
    /// See HwCommand::EnableSynchronization() for more details.
    virtual void Synchronize() = 0;

    /// @brief Check whether synchronization is enabled for the HwCommand.
    /// See HwCommand::EnableSynchronization() for more details.
    /// @return true if synchronization is enabled, false otherwise.
    virtual bool Synchronizable() = 0;

    /// @brief Enable synchronization for the HwCommand. After this function returns true,
    /// Synchronize() can be called on this HwCommand. See HwCommand::Synchronize().
    /// XPU drivers typically provide events (e.g., CUDA events) or fences (e.g., LevelZero fences)
    /// to do fine-grained synchronization. This function can be implemented by creating events
    /// or fences and appending them after launching the HwCommand.
    /// @return Whether the synchronization is supported and has been enabled successfully.
    virtual bool EnableSynchronization() = 0;

    /// @brief Wait until the HwCommand actually become "completed", i.e., executed successfully.
    virtual void Wait() final override;

    /// @brief A callback function that will be called before launching the HwCommand on the
    /// HwQueue. Can do synchronization here. For example, a cuStreamWaitEvent command can wait
    /// here until the CUDA event command is completed, to ensure that cuStreamWaitEvent() will
    /// called (or launched) after the CUDA event is recorded.
    /// NOTE: This function will be called without holding the launch worker lock.
    virtual void BeforeLaunch() {}

    /// @brief Get the handle of the HwCommand.
    /// @return The handle of the HwCommand, which is unique on the system.
    virtual HwCommandHandle GetHandle() {return (uint64_t)this ^ ((uint64_t)GetProcessId() << 48);}

    /// @brief Get the index of the HwCommand submitted to the XQueue. The index starts from 1,
    /// available AFTER the HwCommand has been submitted to the XQueue.
    /// @return The index of the HwCommand.
    int64_t GetIdx() const { return idx_; }

    /// @brief Set the index of the HwCommand.
    /// @param idx The index of the HwCommand to set.
    void SetIdx(int64_t idx) { idx_ = idx; }

    /// @brief Get the handle of the XQueue that the HwCommand is submitted to.
    /// @return The handle of the XQueue. 0 if the HwCommand is not submitted to any XQueue.
    XQueueHandle GetXQueueHandle() const { return xqueue_handle_; }

    /// @brief A callback function that will be called when the HwCommand is submitted to XQueue.
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
