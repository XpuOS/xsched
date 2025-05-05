#pragma once

#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/vpi/hal/driver.h"

#define VPI_ASSERT(cmd) \
    do { \
        VPIStatus res = cmd; \
        if (UNLIKELY(res != VPI_SUCCESS)) {       \
            char msg[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
            const char *name = xsched::vpi::Driver::StatusGetName(res);  \
            xsched::vpi::Driver::GetLastStatusMessage(msg, sizeof(msg)); \
            XERRO("vpi error %d(%s): %s", res, name, msg); \
        } \
    } while (0);
