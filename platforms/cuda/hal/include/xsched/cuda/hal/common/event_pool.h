#pragma once

#include <map>
#include <mutex>
#include <memory>

#include "xsched/utils/pool.h"
#include "xsched/utils/common.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

namespace xsched::cuda
{

class ContextEventPool : public xsched::utils::ObjectPool
{
public:
    ContextEventPool(CUcontext ctx): ctx_(ctx) {}
    virtual ~ContextEventPool() = default;

private:
    virtual void *Create() override;

    const CUcontext ctx_;
};

class CudaEventPool
{
public:
    STATIC_CLASS(CudaEventPool);
    
    static CUevent Pop(CUcontext ctx);
    static void Push(CUcontext ctx, CUevent event);

private:
    static std::mutex mutex_;
    static std::map<CUcontext, std::shared_ptr<ContextEventPool>> pools_;
};

} // namespace xsched::cuda
