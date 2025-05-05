#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/cudla/hal/cudla.h"
#include "xsched/cudla/hal/cudart.h"

#define CUDLA_ASSERT(cmd) \
    do { \
        cudlaStatus result = cmd; \
        if (UNLIKELY(result != cudlaSuccess)) { \
            XERRO("cudla error %d", result);    \
        } \
    } while (0);

#define CUDART_ASSERT(cmd) \
    do { \
        cudaError_t result = cmd; \
        if (UNLIKELY(result != cudaSuccess)) {      \
            XERRO("cuda runtime error %d", result); \
        } \
    } while (0);
