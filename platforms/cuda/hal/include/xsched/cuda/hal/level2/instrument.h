#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include "xsched/utils/common.h"
#include "xsched/cuda/hal/level2/mm.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/cuda_command.h"

namespace xsched::cuda
{

struct InstrumentedKernel
{
    CUfunction func;
    CUdeviceptr entry_point_original;
    CUdeviceptr entry_point_instrumented;
};

enum KernelLaunchType
{
    kKernelLaunchDefault        = 0,
    kKernelLaunchInstrumented   = 1,
    kKernelLaunchResume         = 2,
};

/// @brief InstrumentContext manages the instrumentation of CUDA kernels
/// within a specific CUcontext.
class InstrumentContext
{
public:
    InstrumentContext(CUcontext context);
    ~InstrumentContext();

    static std::shared_ptr<InstrumentContext> GetInstrumentContext(CUcontext context);

    CUstream OpStream() const { return kOpStream; }
    CUresult Launch(std::shared_ptr<CudaKernelCommand> kernel,
                    CUstream stream, KernelLaunchType type);
    void Instrument(std::shared_ptr<CudaKernelCommand> kernel);

private:
    const CUcontext kContext;
    const CUstream kOpStream;

    std::mutex launch_mtx_;
    std::mutex kernel_map_mtx_;
    std::unordered_map<CUfunction, InstrumentedKernel> kernel_map_;

    CUdeviceptr entry_point_resume_ = 0;
    std::unique_ptr<InstrMemAllocator> instr_mem_ = nullptr;
    std::unique_ptr<ResizableBuffer> preempt_buf_default_ = nullptr;

    static void GetCheckInstructions(const void **instr, size_t *size);
    static void GetResumeInstructions(const void **instr, size_t *size);
};

class InstrumentManager
{
public:
    InstrumentManager(CUcontext context, CUstream stream);
    ~InstrumentManager();

    void Deactivate();
    uint64_t Reactivate();

    void Instrument(std::shared_ptr<CudaKernelCommand> kernel);
    void Launch(std::shared_ptr<CudaKernelCommand> kernel, XPreemptLevel level);

private:
    const CUstream kStream;
    uint64_t preempt_idx_ = 0;
    std::unique_ptr<ResizableBuffer> preempt_buf_ = nullptr;
    std::shared_ptr<InstrumentContext> instrument_ctx_ = nullptr;
};

} // namespace xsched::cuda
