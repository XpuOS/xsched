#include <mutex>
#include <cstring>
#include <unordered_map>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/level2/op_stream.h"
#include "xsched/cuda/hal/level2/instrument.h"

#define MAX_DEBUG_BLOCK_SIZE    2048
#define PREEMPT_BUFFER_DEBUG    false

using namespace xsched::cuda;
using namespace xsched::preempt;

InstrumentContext::InstrumentContext(CUcontext context)
    : kContext(context), kOpStream(OpStreamManager::GetOpStream(context))
{
    instr_mem_ = std::make_unique<InstrMemAllocator>();
    preempt_buf_default_ = std::make_unique<ResizableBuffer>();

    // prepare the resume instructions
    size_t resume_size;
    const void *resume_host;
    GetResumeInstructions(&resume_host, &resume_size);
    entry_point_resume_ = instr_mem_->Alloc(resume_size);
    cuXtraInstrMemcpyHtoD(entry_point_resume_,
                          resume_host, resume_size, kOpStream);
}

InstrumentContext::~InstrumentContext()
{

}

std::shared_ptr<InstrumentContext> InstrumentContext::GetInstrumentContext(CUcontext context)
{
    static std::mutex ctx_map_mtx;
    static std::map<CUcontext, std::shared_ptr<InstrumentContext>> ctx_map;

    std::lock_guard<std::mutex> lock(ctx_map_mtx);
    auto it = ctx_map.find(context);
    if (it != ctx_map.end()) return it->second;

    auto ctx = std::make_shared<InstrumentContext>(context);
    ctx_map[context] = ctx;
    return ctx;
}

CUresult InstrumentContext::Launch(std::shared_ptr<CudaKernelCommand> kernel,
                                   CUstream stream, KernelLaunchType type)
{
    char args_buf[28];
    uint64_t *preempt_buf  = (uint64_t *)(args_buf +  0); // 1st arg: preempt buffer addr
    uint64_t *instrumented = (uint64_t *)(args_buf +  8); // 2nd arg: instrumented entry point
    int64_t  *kernel_idx   = (int64_t  *)(args_buf + 16); // 3rd arg: kernel index
    uint32_t *killable     = (uint32_t *)(args_buf + 24); // 4th arg: killable flag

    if (type == kKernelLaunchDefault) {
        size_t block_cnt = kernel->BlockCnt();
        size_t buf_size = 2 * sizeof(uint64_t)
                        + 2 * sizeof(uint32_t) * block_cnt;
        preempt_buf_default_->ExpandTo(buf_size);
    
        // only preempt_buf is useful in this case
        memset(args_buf, 0, sizeof(args_buf));
        *preempt_buf = preempt_buf_default_->DevPtr();

        launch_mtx_.lock();
        cuXtraSetDebuggerParams(kernel->kFunc, args_buf, sizeof(args_buf));
        CUresult ret = kernel->LaunchWrapper(stream);
        launch_mtx_.unlock();
        return ret;
    }

    *preempt_buf  = kernel->preempt_buffer;
    *instrumented = kernel->entry_point_instrumented;
    *kernel_idx   = kernel->GetIdx();
    *killable     = kernel->killable;
    CUdeviceptr entry_point = type == kKernelLaunchResume
                            ? entry_point_resume_ // launch to resume
                            : kernel->entry_point_instrumented;
    
    launch_mtx_.lock();
    cuXtraSetDebuggerParams(kernel->kFunc, args_buf, sizeof(args_buf));
    cuXtraSetEntryPoint(kernel->kFunc, entry_point);
    CUresult ret = kernel->LaunchWrapper(stream);
    cuXtraSetEntryPoint(kernel->kFunc, kernel->entry_point_original);
    launch_mtx_.unlock();

    return ret;
}

void InstrumentContext::Instrument(std::shared_ptr<CudaKernelCommand> kernel)
{
    CUfunction func = kernel->kFunc;

    kernel_map_mtx_.lock();
    auto it = kernel_map_.find(func);
    if (it != kernel_map_.end()) {
        // the kernel has been instrumented
        kernel->entry_point_original = it->second.entry_point_original;
        kernel->entry_point_instrumented = it->second.entry_point_instrumented;
        kernel_map_mtx_.unlock();
        return;
    }
    kernel_map_mtx_.unlock();

    // the kernel has not been instrumented, instrument it
    launch_mtx_.lock();
    // get the original entry point of the kernel
    CUdeviceptr ep_orig = cuXtraGetEntryPoint(func);
    launch_mtx_.unlock();

    kernel_map_mtx_.lock();

    size_t check_size, kernel_size;
    const void *check_host, *kernel_host;
    GetCheckInstructions(&check_host, &check_size);
    cuXtraGetBinary(kContext, func, &kernel_host, &kernel_size, false);

    // allocate memory for the instrumented kernel, return ptr is entry point
    CUdeviceptr ep_inst = instr_mem_->Alloc(kernel_size + check_size);
    // the instrumented kernel starts with the preemption check instructions
    cuXtraInstrMemcpyHtoD(ep_inst, check_host, check_size, kOpStream);
    // followed by the original kernel instructions
    cuXtraInstrMemcpyHtoD(ep_inst + check_size, kernel_host, kernel_size,
                          kOpStream);
    
    // the preemption check instructions will use 32 regs per thread
    size_t reg_cnt = cuXtraGetLocalRegsPerThread(func);
    if (reg_cnt < 32) cuXtraSetLocalRegsPerThread(func, 32);

    // the preemption check instructions will use 1 barrier
    size_t barrier_cnt = cuXtraGetBarrierCnt(func);
    if (barrier_cnt < 1) cuXtraSetBarrierCnt(func, 1);

    // flush instruction cache to take effect
    cuXtraInvalInstrCache(kContext);
    
    // update the instrumented kernel map
    kernel_map_[func] = InstrumentedKernel {
        .func = func,
        .entry_point_original = ep_orig,
        .entry_point_instrumented = ep_inst,
    };

    kernel_map_mtx_.unlock();

    kernel->entry_point_original = ep_orig;
    kernel->entry_point_instrumented = ep_inst;
}


InstrumentManager::InstrumentManager(CUcontext context, CUstream stream): kStream(stream)
{
    instrument_ctx_ = InstrumentContext::GetInstrumentContext(context);
    preempt_buf_ = std::make_unique<ResizableBuffer>();
}

InstrumentManager::~InstrumentManager()
{
    instrument_ctx_ = nullptr;
    preempt_buf_ = nullptr;
}

/* preempt buffer layout: (see also tools/instrument/inject.cu)
 * |<------- uint32 ------->|<--- 32 bits -->|<---- uint64 ----->|<--------- uint32 --------->|<----------- uint32 ---------->|<------ ... ------>|
 * |<-- global_exit_flag -->|<-- reserved -->|<-- preempt_idx -->|<-- exit_flag_of_block_0 -->|<-- restore_flag_of_block_0 -->|<-- block_1 ... -->|
 */
void InstrumentManager::Deactivate()
{
    // set global_exit_flag to 1
    CUDA_ASSERT(Driver::MemsetD32Async(preempt_buf_->DevPtr(), 1, 1,
                                       instrument_ctx_->OpStream()));
}

uint64_t InstrumentManager::Reactivate()
{
    // read preempt_idx from preempt buffer
    CUDA_ASSERT(Driver::MemcpyDtoHAsync_v2(
        &preempt_idx_, preempt_buf_->DevPtr() + sizeof(uint64_t),
        sizeof(uint64_t), instrument_ctx_->OpStream()));
    // clear the header of preempt buffer
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buf_->DevPtr(), 0,
        2 * sizeof(uint64_t), instrument_ctx_->OpStream()));
    CUDA_ASSERT(Driver::StreamSynchronize(instrument_ctx_->OpStream()));

#if PREEMPT_BUFFER_DEBUG
    XINFO("preempt idx: %lu\n", preempt_idx_);
    uint32_t buffer_host[MAX_DEBUG_BLOCK_SIZE * 2 + 4];
    CUDA_ASSERT(Driver::MemcpyDtoHAsync_v2(buffer_host, preempt_buf_->DevPtr(),
        sizeof(buffer_host), instrument_ctx_->OpStream()));
    CUDA_ASSERT(Driver::StreamSynchronize(instrument_ctx_->OpStream()));
    for (size_t i = 0; i < MAX_DEBUG_BLOCK_SIZE; ++i) {
        XINFO("block[%ld]:\t%d,\t%d",
              i, buffer_host[2*i+4], buffer_host[2*i+5]);
    }
#endif

    return preempt_idx_;
}

void InstrumentManager::Launch(std::shared_ptr<CudaKernelCommand> kernel, XPreemptLevel level)
{
    KernelLaunchType launch_type = kKernelLaunchDefault;
    if (level >= kPreemptLevelDeactivate) {
        launch_type = (int64_t)preempt_idx_ == kernel->GetIdx()
                    ? kKernelLaunchResume // the first preempted kernel
                    : kKernelLaunchInstrumented;
    }
    instrument_ctx_->Launch(kernel, kStream, launch_type);
}

void InstrumentManager::Instrument(std::shared_ptr<CudaKernelCommand> kernel)
{
    size_t block_cnt = kernel->BlockCnt();
    size_t buf_size = 2 * sizeof(uint64_t)
                    + 2 * sizeof(uint32_t) * block_cnt;
    preempt_buf_->ExpandTo(buf_size);
    instrument_ctx_->Instrument(kernel);
    kernel->preempt_buffer = preempt_buf_->DevPtr();
}
