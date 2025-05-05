#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"
#include "xsched/cuda/hal/level2/instrument.h"
#include "xsched/cuda/hal/level3/trap.h"

SET_CUDA_QUEUE_LEVEL(1);
using namespace xsched::cuda;

void TrapManager::InstrumentTrapHandler(void *,
                                        CUdeviceptr ,
                                        size_t ,
                                        void *,
                                        CUdeviceptr ,
                                        size_t)
{
    XERRO("Level-3 not implemented for this device");
}

void InstrumentContext::GetResumeInstructions(const void **,
                                              size_t *)
{
    XERRO("Level-2 not implemented for this device");
}

void InstrumentContext::GetCheckInstructions(const void **,
                                             size_t *)
{
    XERRO("Level-2 not implemented for this device");
}
