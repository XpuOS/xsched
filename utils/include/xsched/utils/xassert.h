#pragma once

#include <cstdio>
#include <cstdlib>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"

#define XASSERT(expr, format, ...) \
    do { \
        bool val = static_cast<bool>(expr); \
        if (UNLIKELY(!val)) { \
            XERRO("Assertion failed: " format, ##__VA_ARGS__); \
        } \
    } while (0);
