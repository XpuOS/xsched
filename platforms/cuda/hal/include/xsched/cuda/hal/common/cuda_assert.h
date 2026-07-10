#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
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

#define CUPTI_ASSERT(cmd) \
    do { \
        CUptiResult result = cmd; \
        if (UNLIKELY(result != CUPTI_SUCCESS)) { \
            const char *str; \
            xsched::cuda::PTI::GetResultString(result, &str); \
            XERRO("cupti error %d: %s", result, str); \
        } \
    } while (0);
