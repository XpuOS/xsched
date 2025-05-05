#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/ascend/hal/acl.h"

#define ACL_ASSERT(cmd) \
    do { \
        aclError result = cmd; \
        if (UNLIKELY(result != ACL_SUCCESS)) { \
            XERRO("acl error %d", result); \
        } \
    } while (0);
