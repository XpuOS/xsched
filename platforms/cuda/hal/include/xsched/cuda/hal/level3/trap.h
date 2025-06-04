#pragma once

#include <memory>

#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/level2/mm.h"

namespace xsched::cuda
{

class TarpHandler
{
public:
    TarpHandler() = default;
    virtual ~TarpHandler() = default;
    static std::shared_ptr<TarpHandler> Instance(CUdevice dev);

    virtual size_t GetInjectSize() = 0;
    virtual void Instrument(void *handler_host, CUdeviceptr handler_dev, size_t size,
                            void *inject_host , CUdeviceptr inject_dev) = 0;
};

} // namespace xsched::cuda
