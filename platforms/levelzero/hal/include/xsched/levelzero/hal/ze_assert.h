#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/levelzero/hal/driver.h"

#define ZE_ASSERT(cmd) \
    do { \
        ze_result_t res = cmd; \
        if (UNLIKELY(res != ZE_RESULT_SUCCESS)) { \
            XERRO("levelzero error 0x%x", res);   \
        } \
    } while (0);
