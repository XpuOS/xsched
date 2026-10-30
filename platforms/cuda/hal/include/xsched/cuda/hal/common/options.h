#pragma once

namespace xsched::cuda
{

enum CudaLv3Implementation
{
    kCudaLv3ImplementationTrap,
    kCudaLv3ImplementationTsg,
    kCudaLv3ImplementationChannel,
};

bool GetCudaSingleStreamPerProcessEnabled();
CudaLv3Implementation GetCudaLv3Implementation();

} // namespace xsched::cuda
