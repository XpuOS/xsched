#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include "xsched/utils/common.h"
#include "xsched/cuda/hal/level2/mm.h"
#include "xsched/cuda/hal/level2/guardian.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/cuda_command.h"

namespace xsched::cuda
{

struct InstrumentedKernel
{
    CUfunction func;
    CUdeviceptr entry_point_original;
    CUdeviceptr entry_point_guardian;
};

enum LaunchType
{
    kKernelLaunchOriginal = 0,
    kKernelLaunchGuardian = 1,
    kKernelLaunchResume   = 2,
};

/// @brief InstrumentContext manages the instrumentation of CUDA kernels within a CUcontext.
class InstrumentContext
{
public:
    InstrumentContext(CUcontext ctx);
    ~InstrumentContext() = default;

    CUstream OpStream() const { return op_stream_; }
    static std::shared_ptr<InstrumentContext> Instance(CUcontext ctx);

    void NotifyTrapInstrumented();
    void Instrument(std::shared_ptr<CudaKernelCommand> kernel);
    CUresult Launch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream, LaunchType type);

private:
    const CUcontext kCtx;
    CUstream op_stream_;
    bool trap_instrumented_ = false;
    CUdeviceptr entry_point_resume_ = 0;
    std::shared_ptr<Guardian> guardian_ = nullptr;
    std::unique_ptr<InstrMemAllocator> instr_mem_ = nullptr;
    std::unique_ptr<ResizableBuffer> preempt_buf_ = nullptr;

    std::mutex launch_mtx_;
    std::mutex instrument_mtx_;
    std::unordered_map<CUfunction, InstrumentedKernel> kernels_;
};

class InstrumentManager
{
public:
    InstrumentManager(CUcontext ctx, CUdevice dev);
    ~InstrumentManager() = default;

    void Deactivate();
    uint64_t Reactivate();
    CUstream OpStream() const { return op_stream_; }
    void NotifyTrapInstrumented() { instrument_ctx_->NotifyTrapInstrumented(); }
    void Instrument(std::shared_ptr<CudaKernelCommand> kernel);
    void Launch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream, XPreemptLevel level);

private:
    uint64_t preempt_idx_ = 0;
    CUstream op_stream_ = nullptr;
    CUdeviceptr preempt_buf_ptr_ = 0;
    std::unique_ptr<ResizableBuffer> preempt_buf_ = nullptr;
    std::shared_ptr<InstrumentContext> instrument_ctx_ = nullptr;
};

} // namespace xsched::cuda
