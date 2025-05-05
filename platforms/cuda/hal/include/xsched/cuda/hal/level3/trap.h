#pragma once

#include <map>
#include <mutex>
#include <memory>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

class TrapManager
{
public:
    TrapManager(CUcontext context);
    ~TrapManager() = default;

    void SetTrapHandler();
    void DumpTrapHandler();
    void InterruptContext();

    static std::shared_ptr<TrapManager> GetTrapManager(CUcontext context);

private:
    const CUcontext kContext;
    const CUstream kOpStream;

    size_t trap_handler_size_ = 0;
    CUdeviceptr trap_handler_dev_ = 0;

    static void InstrumentTrapHandler(void *trap_handler_host,
                                      CUdeviceptr trap_handler_dev,
                                      size_t trap_handler_size,
                                      void *extra_instrs_host,
                                      CUdeviceptr extra_instrs_device,
                                      size_t extra_instrs_size);
};

} // namespace xsched::cuda
