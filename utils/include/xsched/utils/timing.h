#pragma once

#include <string>
#include <chrono>
#include <cstdint>
#include <sys/time.h>
#include <sys/resource.h>

#include "xsched/utils/xassert.h"

inline int64_t GetCpuTimeUs(const rusage *usage)
{
    return usage->ru_utime.tv_sec * 1000000 + usage->ru_utime.tv_usec +
           usage->ru_stime.tv_sec * 1000000 + usage->ru_stime.tv_usec;
}

#define EXEC_TIME(time_unit, body) \
({ \
    auto begin = std::chrono::system_clock::now(); \
    do { body; } while (0);                        \
    auto end = std::chrono::system_clock::now();   \
    (int64_t)std::chrono::duration_cast<std::chrono::time_unit> \
        (end - begin).count(); \
})

#define EXEC_US_CPU(body) \
({ \
    rusage begin, end; \
    XASSERT(getrusage(RUSAGE_SELF, &begin) == 0, "getrusage failed"); \
    do { body; } while (0); \
    XASSERT(getrusage(RUSAGE_SELF, &end) == 0, "getrusage failed");   \
    GetCpuTimeUs(&end) - GetCpuTimeUs(&begin); \
})

#define CPU_UTIL(body) \
({ \
    int64_t cpu_us = 0; \
    int64_t total_us = EXEC_TIME(microseconds, { \
        cpu_us = EXEC_US_CPU(body); \
    }); \
    (double)cpu_us / total_us; \
})

namespace xsched::utils
{

class Timer
{
public:
    Timer(const std::string &name): kName(name) {}
    ~Timer();

    void RecordBegin();
    void RecordEnd();

    void LogResults();
    int64_t GetAvgNs();
    int64_t GetAvgNsCpu();

private:
    const std::string kName;

    int64_t total_ns_ = 0;
    int64_t total_cpu_ns_ = 0;
    int64_t total_cnt_ = 0;

    rusage usage_begin_, usage_end_;
    std::chrono::system_clock::time_point begin_;
    std::chrono::system_clock::time_point end_;
};

class Accumulator
{
public:
    void Start();
    void Stop();
    void Reset();
    int64_t GetAccumulatedNs();

private:
    int64_t accumulated_ns_ = 0;
    bool started_ = false;
    std::chrono::system_clock::time_point begin_;
};

} // namespace xsched::utils
