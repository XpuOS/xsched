#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/opencl/hal/cl.h"

#define OCL_ASSERT(cmd) \
    do { \
        cl_int res = cmd; \
        if (UNLIKELY(res != CL_SUCCESS)) { \
            XERRO("opencl error %d", res); \
        } \
    } while (0);
