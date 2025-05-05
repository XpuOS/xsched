#include "cudart_utils.h"

uint64_t GetClockRate()
{
    cudaDeviceProp prop;
    CUDART_ASSERT(cudaGetDeviceProperties(&prop, 0));
    return prop.clockRate * 1000;
}

uint64_t ConvertClockCnt(uint64_t microseconds)
{
    static uint64_t clock_rate = GetClockRate();
    return (microseconds * clock_rate) / 1000000;
}
