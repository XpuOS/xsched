#include "xsched/xqueue.h"
#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/preempt/hal/hw_command.h"

using namespace xsched::preempt;

HwCommand::HwCommand(XCommandProperties props): XCommand(kCommandTypeHardware, props)
{
    this->AddStateListener([this](XCommandState state) {
        if (state < kCommandStateCompleted) return;
        auto lock = this->GetLock();
        this->xqueue_ = nullptr; // break reference and allow XQueue to be destroyed
    });
}

void HwCommand::Wait()
{
    auto lock = this->GetLock();
    XQueueHandle xqueue_handle = xqueue_handle_;
    std::shared_ptr<XQueue> xqueue = xqueue_;
    lock.unlock();

    if (xqueue_handle == 0) {
        // The HwCommand has not been submitted to an XQueue,
        // so we need to synchronize with the hardware.
        XASSERT(this->Synchronizable(),
                "The HwCommand being waited should either be Synchronizable "
                "or have been submitted to an XQueue.");
        return this->Synchronize();
    }

    if (xqueue == nullptr) {
        // The HwCommand has been submitted to an XQueue, but the XQueue pointer
        // has been released. This could only happen when the HwCommand has been completed.
        XASSERT(this->GetState() >= kCommandStateCompleted,
                "The HwCommand should have been completed before waiting");
        return;
    }

    xqueue->Wait(std::static_pointer_cast<HwCommand>(shared_from_this()));
}

void HwCommand::OnSubmit(std::shared_ptr<XQueue> xqueue)
{
    XASSERT(xqueue != nullptr, "XQueue is nullptr");
    auto lock = this->GetLock();
    xqueue_ = xqueue;
    xqueue_handle_ = xqueue->GetHandle();
}

std::mutex HwCommandManager::mtx_;
std::unordered_map<HwCommandHandle, std::shared_ptr<HwCommand>> HwCommandManager::hw_cmds_;

HwCommandHandle HwCommandManager::Add(std::shared_ptr<HwCommand> hw_cmd)
{
    std::lock_guard<std::mutex> lock(mtx_);
    HwCommandHandle hw_cmd_h = hw_cmd->GetHandle();
    hw_cmds_[hw_cmd_h] = hw_cmd;
    return hw_cmd_h;
}

std::shared_ptr<HwCommand> HwCommandManager::Del(HwCommandHandle hw_cmd)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = hw_cmds_.find(hw_cmd);
    if (it == hw_cmds_.end()) return nullptr;
    std::shared_ptr<HwCommand> hw_cmd_shptr = it->second;
    hw_cmds_.erase(it);
    return hw_cmd_shptr;
}

std::shared_ptr<HwCommand> HwCommandManager::Get(HwCommandHandle hw_cmd)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = hw_cmds_.find(hw_cmd);
    if (it == hw_cmds_.end()) return nullptr;
    return it->second;
}

EXPORT_C_FUNC XResult HwCommandCreateCallback(HwCommandHandle *hw_cmd,
                                              LaunchCallback launch, void *data)
{
    if (hw_cmd == nullptr) return kXSchedErrorInvalidValue;
    std::shared_ptr<HwCommand> hw_cmd_shptr = std::make_shared<HwCallbackCommand>(launch, data);
    *hw_cmd = HwCommandManager::Add(hw_cmd_shptr);
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult HwCommandDestroy(HwCommandHandle hw_cmd)
{
    std::shared_ptr<HwCommand> hw_cmd_shptr = HwCommandManager::Del(hw_cmd);
    if (hw_cmd_shptr == nullptr) {
        XWARN("HwCommand with handle 0x%lx does not exist or not registered", hw_cmd);
        return kXSchedErrorNotFound;
    }
    return kXSchedSuccess;
}
