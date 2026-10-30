#include "cudart_utils.h"

uint64_t GetClockRate()
{
    int clock_rate_khz;
    CUDART_ASSERT(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
    return static_cast<uint64_t>(clock_rate_khz) * 1000;
}

uint64_t ConvertClockCnt(uint64_t microseconds)
{
    static uint64_t clock_rate = GetClockRate();
    return (microseconds * clock_rate) / 1000000;
}
