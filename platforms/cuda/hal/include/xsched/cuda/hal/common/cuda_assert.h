#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/driver.h"

#define CUDA_ASSERT(cmd) \
    do { \
        CUresult result = cmd; \
        if (UNLIKELY(result != CUDA_SUCCESS)) { \
            const char *str; \
            xsched::cuda::Driver::GetErrorString(result, &str); \
            XERRO("cuda error %d: %s", result, str); \
        } \
    } while (0);
