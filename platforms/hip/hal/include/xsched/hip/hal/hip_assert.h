#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/hip/hal/driver.h"

#define HIP_ASSERT(cmd) \
    do { \
        hipError_t result = cmd; \
        if (UNLIKELY(result != hipSuccess)) { \
            const char *str = xsched::hip::Driver::GetErrorString(result); \
            XERRO("hip error %d: %s", result, str); \
        } \
    } while (0);
