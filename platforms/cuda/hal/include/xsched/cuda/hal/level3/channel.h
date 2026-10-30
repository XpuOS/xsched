#pragma once

#include <mutex>
#include "xsched/cuda/hal/level3/channel_control.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"

namespace xsched::cuda
{

// Experimental Linux RM backend. One managed queue owns the context's compute group.
class CudaQueueLv3Channel : public CudaQueueLv1
{
public:
    static std::shared_ptr<preempt::HwQueue> Create(CUstream stream);
    static CUresult DestroyContext(CUcontext context, bool legacy);
    static CUresult ResetPrimaryContext(CUdevice device, bool legacy);
    ~CudaQueueLv3Channel() override;
    void Deactivate() override;
    void Interrupt() override;
    void Restore(const preempt::CommandLog &) override;
    void Reactivate(const preempt::CommandLog &) override;
    void SynchronizeOnSuspend() override;
    void OnPreemptLevelChange(XPreemptLevel level) override;
    bool SupportDynamicLevel() override { return true; }
    XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }
    bool SupportResumeDropCommands() const override { return false; }
    CUcontext GetContext() const { return context_; }

private:
    using Control = channel::Control;
    CudaQueueLv3Channel(CUstream stream, Control control, int fd,
                       unsigned client, unsigned subdevice, unsigned group);
    Control control_ = nullptr;
    int fd_ = -1;
    unsigned client_ = 0;
    unsigned subdevice_ = 0;
    unsigned group_ = 0;
    bool paused_ = false;
    std::mutex mutex_;
    void Pause();
    void Resume();
};

} // namespace xsched::cuda
