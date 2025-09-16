#include <mutex>
#include <cstring>
#include <unordered_map>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/level2/instrument.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

InstrumentContext::InstrumentContext(CUcontext ctx): kCtx(ctx)
{
    // check current context
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == ctx,
            "create InstrumentContext failed: current context (%p) does not match context (%p)",
            current_ctx, ctx);

    // create operation stream with highest priority
    int lp, hp;
    CUDA_ASSERT(Driver::CtxGetStreamPriorityRange(&lp, &hp));
    CUDA_ASSERT(Driver::StreamCreateWithPriority(&op_stream_, 0, hp));

    CUdevice dev;
    CUDA_ASSERT(Driver::CtxGetDevice(&dev));
    guardian_ = Guardian::Instance(dev);
    instr_mem_ = std::make_unique<InstrMemAllocator>(kCtx, dev);
    preempt_buf_ = std::make_unique<ResizableBuffer>(dev);
    // clear the preempt buffer
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buf_->Ptr(), 0, preempt_buf_->Size(), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));

    // prepare the resume instructions
    size_t resume_size;
    const void *resume_host;
    guardian_->GetResumeInstructions(&resume_host, &resume_size);
    entry_point_resume_ = instr_mem_->Alloc(resume_size);
    cuXtraInstrMemcpyHtoD(entry_point_resume_, resume_host, resume_size, op_stream_);
}

std::shared_ptr<InstrumentContext> InstrumentContext::Instance(CUcontext ctx)
{
    static std::mutex ctx_mtx;
    static std::map<CUcontext, std::shared_ptr<InstrumentContext>> ctx_map;

    std::lock_guard<std::mutex> lock(ctx_mtx);
    auto it = ctx_map.find(ctx);
    if (it != ctx_map.end()) return it->second;

    auto instr_ctx = std::make_shared<InstrumentContext>(ctx);
    ctx_map[ctx] = instr_ctx;
    return instr_ctx;
}

CUresult InstrumentContext::Launch(std::shared_ptr<CudaKernelCommand> kernel,
                                   CUstream stream, LaunchType type)
{
    char args_buf[28];
    uint64_t *preempt_buf = (uint64_t *)(args_buf +  0); // 1st arg: preempt buffer addr
    uint64_t *guardian    = (uint64_t *)(args_buf +  8); // 2nd arg: guardian entry point
    int64_t  *kernel_idx  = (int64_t  *)(args_buf + 16); // 3rd arg: kernel index
    uint32_t *killable    = (uint32_t *)(args_buf + 24); // 4th arg: killable flag

    if (type == kKernelLaunchOriginal) {
        size_t block_cnt = kernel->BlockCnt();
        size_t buf_size = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t) * block_cnt;
        preempt_buf_->ExpandTo(buf_size, op_stream_);

        // only preempt_buf is useful in this case
        memset(args_buf, 0, sizeof(args_buf));
        *preempt_buf = preempt_buf_->Ptr(); // use default (empty) preempt buffer

        launch_mtx_.lock();
        cuXtraSetDebuggerParams(kernel->kFunc, args_buf, sizeof(args_buf));
        CUresult ret = kernel->LaunchWrapper(stream);
        launch_mtx_.unlock();
        return ret;
    }

    *preempt_buf = kernel->preempt_buffer;
    *guardian    = kernel->entry_point_guardian;
    *kernel_idx  = kernel->GetIdx();
    *killable    = kernel->killable;
    CUdeviceptr entry_point = type == kKernelLaunchResume
                            ? entry_point_resume_ // launch to resume
                            : kernel->entry_point_guardian;
    
    launch_mtx_.lock();
    cuXtraSetDebuggerParams(kernel->kFunc, args_buf, sizeof(args_buf));
    cuXtraSetEntryPoint(kernel->kFunc, entry_point);
    CUresult ret = kernel->LaunchWrapper(stream);
    cuXtraSetEntryPoint(kernel->kFunc, kernel->entry_point_original);
    launch_mtx_.unlock();

    return ret;
}

void InstrumentContext::NotifyTrapInstrumented()
{
    std::lock_guard<std::mutex> lock(launch_mtx_);
    trap_instrumented_ = true;
}

void InstrumentContext::Instrument(std::shared_ptr<CudaKernelCommand> kernel)
{
    CUfunction func = kernel->kFunc;

    {
        std::lock_guard<std::mutex> lock(instrument_mtx_);
        auto it = kernels_.find(func);
        if (it != kernels_.end()) {
            // the kernel has been instrumented
            kernel->entry_point_original = it->second.entry_point_original;
            kernel->entry_point_guardian = it->second.entry_point_guardian;
            return;
        }
    }

    // the kernel has not been instrumented, instrument it
    launch_mtx_.lock();
    // get the original entry point of the kernel
    CUdeviceptr ep_orig = cuXtraGetEntryPoint(func);
    launch_mtx_.unlock();

    instrument_mtx_.lock();

    size_t check_size, kernel_size;
    const void *check_host, *kernel_host;
    guardian_->GetGuardianInstructions(&check_host, &check_size);
    cuXtraGetBinary(kCtx, func, &kernel_host, &kernel_size, false);

    // allocate memory for the instrumented kernel, return ptr is entry point
    CUdeviceptr ep_inst = instr_mem_->Alloc(kernel_size + check_size);
    // the instrumented kernel starts with the guardian instructions
    cuXtraInstrMemcpyHtoD(ep_inst, check_host, check_size, op_stream_);
    // followed by the original kernel instructions
    cuXtraInstrMemcpyHtoD(ep_inst + check_size, kernel_host, kernel_size, op_stream_);
    
    // the guardian instructions will use 32 regs per thread
    size_t reg_cnt = cuXtraGetLocalRegsPerThread(func);
    if (reg_cnt < 32) cuXtraSetLocalRegsPerThread(func, 32);

    // the guardian instructions will use 1 barrier
    size_t barrier_cnt = cuXtraGetBarrierCnt(func);
    if (barrier_cnt < 1) cuXtraSetBarrierCnt(func, 1);

    // flush instruction cache to take effect
    cuXtraInvalInstrCache(kCtx);
    
    // update the instrumented kernel map
    kernels_[func] = InstrumentedKernel {
        .func = func,
        .entry_point_original = ep_orig,
        .entry_point_guardian = ep_inst,
    };

    instrument_mtx_.unlock();

    kernel->entry_point_original = ep_orig;
    kernel->entry_point_guardian = ep_inst;
}


InstrumentManager::InstrumentManager(CUcontext ctx, CUdevice dev)
{
    preempt_buf_ = std::make_unique<ResizableBuffer>(dev);
    preempt_buf_ptr_ = preempt_buf_->Ptr();
    instrument_ctx_ = InstrumentContext::Instance(ctx);
    op_stream_ = instrument_ctx_->OpStream();
}

/* preempt buffer layout: (see also tools/instrument/inject.cu)
 * |<------- uint32 ------->|<--- 32 bits -->|<---- uint64 ----->|<--------- uint32 --------->|<----------- uint32 ---------->|<------ ... ------>|
 * |<-- global_exit_flag -->|<-- reserved -->|<-- preempt_idx -->|<-- exit_flag_of_block_0 -->|<-- restore_flag_of_block_0 -->|<-- block_1 ... -->|
 */
void InstrumentManager::Deactivate()
{
    // set global_exit_flag to 1
    CUDA_ASSERT(Driver::MemsetD32Async(preempt_buf_ptr_, 1, 1, op_stream_));
}

uint64_t InstrumentManager::Reactivate()
{
    // read preempt_idx from preempt buffer
    CUDA_ASSERT(Driver::MemcpyDtoHAsync_v2(&preempt_idx_, preempt_buf_ptr_ + sizeof(uint64_t),
                                           sizeof(uint64_t), op_stream_));
    // clear the header of preempt buffer
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buf_ptr_, 0, 2 * sizeof(uint64_t), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));

#define MAX_DEBUG_BLOCK_SIZE    1024
#define PREEMPT_BUFFER_DEBUG    false

#if PREEMPT_BUFFER_DEBUG
    XINFO("preempt idx: %" FMT_64U, preempt_idx_);
    uint32_t buffer_host[MAX_DEBUG_BLOCK_SIZE * 2 + 4];
    CUDA_ASSERT(Driver::MemcpyDtoHAsync_v2(buffer_host, preempt_buf_ptr_,
                                           sizeof(buffer_host), op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));
    for (size_t i = 0; i < MAX_DEBUG_BLOCK_SIZE; ++i) {
        XINFO("block[%ld]:\t%d,\t%d", i, buffer_host[2*i+4], buffer_host[2*i+5]);
    }
#endif

    return preempt_idx_;
}

void InstrumentManager::Launch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream, XPreemptLevel level)
{
    LaunchType launch_type = kKernelLaunchOriginal;
    if (level >= kPreemptLevelDeactivate) {
        launch_type = (int64_t)preempt_idx_ == kernel->GetIdx()
                    ? kKernelLaunchResume // the first preempted kernel
                    : kKernelLaunchGuardian;
    }
    instrument_ctx_->Launch(kernel, stream, launch_type);
}

void InstrumentManager::Instrument(std::shared_ptr<CudaKernelCommand> kernel)
{
    size_t block_cnt = kernel->BlockCnt();
    size_t buf_size = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t) * block_cnt;
    preempt_buf_->ExpandTo(buf_size, op_stream_);
    instrument_ctx_->Instrument(kernel);
    kernel->preempt_buffer = preempt_buf_->Ptr();
}
