#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/corex/hal/common/cuda.h"
#include "xsched/corex/hal/common/cudart.h"

#define COREX_ASSERT(cmd) \
    do { \
        CUresult result = cmd; \
        if (UNLIKELY(result != CUDA_SUCCESS)) { \
            XERRO("corex driver error %d", result); \
        } \
    } while (0)

#define COREXRT_ASSERT(cmd) \
    do { \
        cudaError_t result = cmd; \
        if (UNLIKELY(result != cudaSuccess)) { \
            XERRO("corex runtime error %d", result); \
        } \
    } while (0)
