#include "xsched/cuda/hal/level2/guardian.h"
#include "xsched/cuda/hal/level3/trap.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/arch/sm35.h"
#include "xsched/cuda/hal/arch/sm70.h"
#include "xsched/cuda/hal/arch/sm86.h"
// NEW_CUDA_ARCH: New CUDA architecture support goes here

using namespace xsched::cuda;
using namespace xsched::preempt;

static int32_t GetArch(CUdevice dev)
{
    // see https://developer.nvidia.com/cuda-legacy-gpus
    // see https://developer.nvidia.com/cuda-gpus
    int major, minor;
    CUDA_ASSERT(Driver::DeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CUDA_ASSERT(Driver::DeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    return major * 10 + minor;
}

std::shared_ptr<Guardian> Guardian::Instance(CUdevice dev)
{
    switch (GetArch(dev)) {
    case 35:
        return std::make_shared<GuardianSM35>();
    case 70:
        return std::make_shared<GuardianSM70>();
    case 86:
        return std::make_shared<GuardianSM86>();
    // NEW_CUDA_ARCH: New CUDA architecture support goes here
    default:
        return nullptr;
    }
}

std::shared_ptr<TarpHandler> TarpHandler::Instance(CUdevice dev)
{
    switch (GetArch(dev)) {
    case 70:
        return std::make_shared<TarpHandlerSM70>();
    case 86:
        return std::make_shared<TarpHandlerSM86>();
    // NEW_CUDA_ARCH: New CUDA architecture support goes here
    default:
        return nullptr;
    }
}

std::shared_ptr<HwQueue> xsched::cuda::CudaQueueCreate(CUstream stream)
{
#if defined(_WIN32)
    return std::make_shared<CudaQueueLv1>(stream);
#endif

    if (GetCudaLv3Implementation() == kCudaLv3ImplementationTsg) {
        return std::make_shared<CudaQueueLv3Tsg>(stream);
    }

    CUdevice dev;
    CUcontext stream_ctx;
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::StreamGetCtx(stream, &stream_ctx));
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == stream_ctx,
            "create CudaQueue failed: current context (%p) does not match stream context (%p)",
            current_ctx, stream_ctx);
    CUDA_ASSERT(Driver::CtxGetDevice(&dev));

    switch (GetArch(dev)) {
    case 35: // Kepler: K20, K40, GTX TITAN
        return std::make_shared<CudaQueueLv2>(stream);
    case 70: // Volta: V100, GV100
    case 86: // Ampere: RTX3050 - RTX 3090 Ti
        return std::make_shared<CudaQueueLv3Trap>(stream);
    // NEW_CUDA_ARCH: New CUDA architecture support goes here
    default:
        return std::make_shared<CudaQueueLv1>(stream);
    }
}

CUresult xsched::cuda::DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream)
{
#if defined(_WIN32)
    return CudaQueueLv1::DirectLaunch(kernel, stream);
#endif

    if (GetCudaLv3Implementation() == kCudaLv3ImplementationTsg) {
        return CudaQueueLv3Tsg::DirectLaunch(kernel, stream);
    }

    CUdevice dev;
    CUcontext stream_ctx;
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::StreamGetCtx(stream, &stream_ctx));
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == stream_ctx,
            "direct launch kernel failed: current context (%p) does not match stream context (%p)",
            current_ctx, stream_ctx);
    CUDA_ASSERT(Driver::CtxGetDevice(&dev));

    switch (GetArch(dev)) {
    case 35:
        return CudaQueueLv2::DirectLaunch(kernel, current_ctx, stream);
    case 70:
    case 86:
        return CudaQueueLv3Trap::DirectLaunch(kernel, current_ctx, stream);
    // NEW_CUDA_ARCH: New CUDA architecture support goes here
    default:
        return CUDA_ERROR_NOT_SUPPORTED;
    }
}
