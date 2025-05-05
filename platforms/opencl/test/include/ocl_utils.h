#pragma once

#include <CL/cl.h>

#include "xsched/utils.h"

#define OCL_ASSERT(cmd) \
    do { \
        cl_int res = cmd; \
        if (UNLIKELY(res != CL_SUCCESS)) { \
            XERRO("opencl error %d", res); \
        } \
    } while (0);
