#pragma once

#include "xsched/types.h"
#include "xsched/utils/log.h"

#define XSCHED_ASSERT(cmd) \
    do { \
        XResult result = cmd; \
        if (UNLIKELY(result != kXSchedSuccess)) { \
            XERRO("xsched error %d", result); \
        } \
    } while (0);
