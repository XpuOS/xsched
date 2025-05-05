#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <cuda_runtime.h>

#include "cudart_utils.h"
#include "xsched/utils.h"
#include "xsched/xqueue.h"
#include "xsched/cuda/hal.h"

using namespace xsched::utils;

#define TEST_CNT            100
#define BLOCK_CNT           64
#define THREAD_CNT          64
#define SLEEP_US            800
#define BLOCK_TIME_MAX_US   2000

__device__ __forceinline__ void Wait(uint64_t clock_cnt)
{
    if (clock_cnt == 0) return;
    uint64_t elapsed = 0;
    uint64_t start = clock64();
    while (elapsed < clock_cnt) { elapsed = clock64() - start; }
}

__global__ void WaitFlag(int32_t *flag)
{
    while (*flag == 0) { Wait(1); }
}

__global__ void Sleep(uint64_t clock_cnt)
{
    Wait(clock_cnt);
}

void TestBlockPreempt(uint64_t us, cudaStream_t stream, XQueueHandle xq)
{
    LoopRunner runner;
    DataProcessor<int64_t> preempt_latency;
    DataProcessor<int64_t> restore_latency;
    uint64_t clock_cnt = ConvertClockCnt(us);

    runner.Start([&]()->void {
        for (int i = 0; i < 64; ++i) {
            Sleep<<<BLOCK_CNT, THREAD_CNT, 0, stream>>>(clock_cnt);
        }
        CUDART_ASSERT(cudaStreamSynchronize(stream));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (size_t i = 0; i < 3 * TEST_CNT; ++i) {
        int64_t sleep_us = rand() % SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t preempt_ns = EXEC_TIME(nanoseconds, {
            XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
        });
        sleep_us = (rand() % SLEEP_US) + SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t restore_ns = EXEC_TIME(nanoseconds, {
            XQueueResume(xq, kQueueResumeFlagNone);
        });

        // warmup and cooldown
        if (i < TEST_CNT && i >= 2 * TEST_CNT) continue;
        preempt_latency.Add(preempt_ns);
        restore_latency.Add(restore_ns);
    }

    runner.Stop();

    double restore_avg = restore_latency.Avg() / 1000.0;
    double preempt_avg = preempt_latency.Avg() / 1000.0;
    double p50 = preempt_latency.Percentile(0.50) / 1000.0;
    double p95 = preempt_latency.Percentile(0.95) / 1000.0;
    double p99 = preempt_latency.Percentile(0.99) / 1000.0;

    printf("%lu %.3f %.3f %.3f %.3f\n", us, preempt_avg, p50, p95, p99);
    fflush(stdout);

    XINFO("[RESULT] block time: %lu us, restore avg: %.2f", us, restore_avg);
    XINFO("[RESULT] block time: %lu us, preempt avg: %.2f", us, preempt_avg);
    XINFO("[RESULT] block time: %lu us, preempt p50: %.2f", us, p50);
    XINFO("[RESULT] block time: %lu us, preempt p95: %.2f", us, p95);
    XINFO("[RESULT] block time: %lu us, preempt p99: %.2f", us, p99);
}

void TestDeadLoopPreempt(cudaStream_t stream, XQueueHandle xq)
{
    LoopRunner runner;
    DataProcessor<int64_t> preempt_latency;
    DataProcessor<int64_t> restore_latency;

    int32_t *flag;
    CUDART_ASSERT(cudaMallocManaged(&flag, sizeof(int32_t)));
    *flag = 0;

    runner.Start([&]()->void {
        for (int i = 0; i < 64; ++i) {
            WaitFlag<<<BLOCK_CNT, THREAD_CNT, 0, stream>>>(flag);
        }
        CUDART_ASSERT(cudaStreamSynchronize(stream));
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (size_t i = 0; i < TEST_CNT; ++i) {
        int64_t sleep_us = rand() % SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t preempt_ns = EXEC_TIME(nanoseconds, {
            XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue);
        });
        sleep_us = (rand() % SLEEP_US) + SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t restore_ns = EXEC_TIME(nanoseconds, {
            XQueueResume(xq, kQueueResumeFlagNone);
        });

        // warmup and cooldown
        if (i < TEST_CNT && i >= 2 * TEST_CNT) continue;
        preempt_latency.Add(preempt_ns);
        restore_latency.Add(restore_ns);
    }
    *flag = 1;

    double restore_avg = restore_latency.Avg() / 1000.0;
    double preempt_avg = preempt_latency.Avg() / 1000.0;
    double p50 = preempt_latency.Percentile(0.50) / 1000.0;
    double p95 = preempt_latency.Percentile(0.95) / 1000.0;
    double p99 = preempt_latency.Percentile(0.99) / 1000.0;

    printf("inf %.3f %.3f %.3f %.3f\n", preempt_avg, p50, p95, p99);
    fflush(stdout);
    XINFO("[RESULT] dead loop, restore avg: %.2f", restore_avg);
    XINFO("[RESULT] dead loop, preempt avg: %.2f", preempt_avg);
    XINFO("[RESULT] dead loop, preempt p50: %.2f", p50);
    XINFO("[RESULT] dead loop, preempt p95: %.2f", p95);
    XINFO("[RESULT] dead loop, preempt p99: %.2f", p99);

    runner.Stop();
}

int main()
{
    srand(time(NULL));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    printf("kernel_time avg p50 p90 p99\n");
    fflush(stdout);

    XQueueHandle xq;
    HwQueueHandle hwq;
    CudaQueueCreate(&hwq, stream);
    XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone);
    XQueueSetLaunchConfig(xq, 8, 4);

    for (size_t i = 100; i <= BLOCK_TIME_MAX_US; i += 100) {
        TestBlockPreempt(i, stream, xq);
    }

    TestDeadLoopPreempt(stream, xq);
}
