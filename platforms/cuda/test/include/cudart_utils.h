#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "xsched/utils.h"

#define CUDART_ASSERT(cmd) \
    do { \
        cudaError_t result = cmd; \
        if (UNLIKELY(result != cudaSuccess)) { \
            XERRO("cuda runtime error %d", result); \
        } \
    } while (0);

/// @brief Get GPU clock frequency in hertz.
uint64_t GetClockRate();
uint64_t ConvertClockCnt(uint64_t microseconds);
