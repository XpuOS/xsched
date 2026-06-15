#include "xsched/utils/log.h"
#include "xsched/utils/env.h"
#include "xsched/protocol/def.h"
#include "xsched/cuda/hal/common/options.h"

bool xsched::cuda::GetCudaSingleStreamPerProcessEnabled()
{
    static const bool sspp = GetEnvOption(XSCHED_CUDA_SINGLE_STREAM_PER_PROCESS_ENV_NAME, false);
    return sspp;
}

xsched::cuda::CudaLv3Implementation xsched::cuda::GetCudaLv3Implementation()
{
    static const CudaLv3Implementation impl = []()->CudaLv3Implementation {
        const std::string str = GetEnv(XSCHED_CUDA_LV3_IMPL_ENV_NAME);
        if (str.empty()) return kCudaLv3ImplementationTrap;
        if (str == "TSG") return kCudaLv3ImplementationTsg;
        XWARN("unknown CUDA Level-3 implementation: %s, use default (TRAP)", str.c_str());
        return kCudaLv3ImplementationTrap;
    }();
    return impl;
}
