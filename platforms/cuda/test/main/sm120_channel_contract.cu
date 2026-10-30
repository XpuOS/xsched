#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
#define CK(x) REQUIRE((x) == 0)
constexpr unsigned Threads = 128, Limit = 100000, CopyWords = 2 * 1024 * 1024;

__global__ void Work(volatile unsigned *progress, volatile unsigned *stop,
                     unsigned *counts, unsigned long long *result)
{
    __shared__ unsigned done;
    unsigned long long sum = 0;
    for (unsigned i = 0; i < Limit; ++i) {
        if (threadIdx.x == 0) done = *stop;
        __syncthreads();
        if (done) break;
        auto begin = clock64();
        while (clock64() - begin < 100000ULL) {}
        sum += (unsigned long long)(i + 1) * (threadIdx.x + 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counts, 1u);
            *progress = i + 1;
            __threadfence_system();
        }
    }
    result[threadIdx.x] = sum;
}

struct Task
{
    CUcontext ctx;
    cudaStream_t stream, companion, copy_stream, dependent;
    HwQueueHandle hwq;
    XQueueHandle xq;
    volatile unsigned *host;
    unsigned *mapped, *counts, *copy_dev, *copy_src, *copy_dst;
    unsigned long long *result, *copied;
    cudaEvent_t kernel_done, copy_done, dependent_done;
    unsigned streams;

    void Create(unsigned n)
    {
        streams = n;
#if CUDA_VERSION >= 13000
        CK(cuCtxCreate(&ctx, nullptr, 0, 0));
#else
        CK(cuCtxCreate(&ctx, 0, 0));
#endif
        CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&companion, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&dependent, cudaStreamNonBlocking));
        CK(CudaQueueCreate(&hwq, stream));
        CK(XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone));
        CK(cudaHostAlloc((void **)&host, 3 * sizeof(unsigned), cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer(&mapped, (void *)host, 0));
        host[0] = 0;
        host[1] = 0;
        host[2] = 0;
        CK(cudaMalloc(&counts, 2 * sizeof(unsigned)));
        CK(cudaMalloc(&result, 2 * Threads * sizeof(*result)));
        CK(cudaMallocHost(&copied, 2 * Threads * sizeof(*copied)));
        CK(cudaMalloc(&copy_dev, CopyWords * sizeof(unsigned)));
        CK(cudaMallocHost(&copy_src, CopyWords * sizeof(unsigned)));
        CK(cudaMallocHost(&copy_dst, CopyWords * sizeof(unsigned)));
        for (unsigned i = 0; i < CopyWords; ++i) copy_src[i] = i * 17 + 3;
        CK(cudaEventCreateWithFlags(&kernel_done, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&dependent_done, cudaEventDisableTiming));
        CK(cudaMemset(counts, 0, 2 * sizeof(unsigned)));
        CK(cudaDeviceSynchronize());
        // Exercise the copy path before suspension, including lazy engine setup.
        CK(cudaMemcpyAsync(copy_dev, copy_src, CopyWords * sizeof(unsigned), cudaMemcpyHostToDevice, copy_stream));
        CK(cudaMemcpyAsync(copy_dst, copy_dev, CopyWords * sizeof(unsigned), cudaMemcpyDeviceToHost, copy_stream));
        CK(cudaStreamSynchronize(copy_stream));
    }
    void Launch()
    {
        CK(cuCtxSetCurrent(ctx));
        Work<<<1, Threads, 0, stream>>>(mapped, mapped + 2, counts, result);
        if (streams == 2)
            Work<<<1, Threads, 0, companion>>>(mapped + 1, mapped + 2, counts + 1, result + Threads);
        CK(cudaGetLastError());
        CK(cudaEventRecord(kernel_done, stream));
        CK(cudaStreamWaitEvent(dependent, kernel_done));
        CK(cudaMemcpyAsync(copied, result, Threads * sizeof(*result), cudaMemcpyDeviceToHost, dependent));
        CK(cudaEventRecord(dependent_done, dependent));
    }
    void CheckAndDestroy()
    {
        CK(cuCtxSetCurrent(ctx));
        CK(cudaStreamSynchronize(stream));
        CK(cudaStreamSynchronize(companion));
        CK(cudaStreamSynchronize(dependent));
        CK(cudaStreamSynchronize(copy_stream));
        REQUIRE(std::memcmp(copy_src, copy_dst, CopyWords * sizeof(unsigned)) == 0);
        REQUIRE(cudaEventQuery(dependent_done) == cudaSuccess);
        unsigned n[2];
        CK(cudaMemcpy(n, counts, sizeof(n), cudaMemcpyDeviceToHost));
        for (unsigned t = 0; t < Threads; ++t)
            REQUIRE(copied[t] == (unsigned long long)n[0] * (n[0] + 1) / 2 * (t + 1));
        CK(cudaMemcpy(copied, result, streams * Threads * sizeof(*result), cudaMemcpyDeviceToHost));
        for (unsigned s = 0; s < streams; ++s) {
            REQUIRE(n[s] == host[s] && n[s] > 0 && n[s] < Limit);
            for (unsigned t = 0; t < Threads; ++t)
                REQUIRE(copied[s * Threads + t] ==
                        (unsigned long long)n[s] * (n[s] + 1) / 2 * (t + 1));
        }
        CK(XQueueDestroy(xq));
        CK(HwQueueDestroy(hwq));
        CK(cudaEventDestroy(kernel_done));
        CK(cudaEventDestroy(copy_done));
        CK(cudaEventDestroy(dependent_done));
        CK(cudaFree(counts)); CK(cudaFree(result)); CK(cudaFree(copy_dev));
        CK(cudaFreeHost(copied)); CK(cudaFreeHost(copy_src)); CK(cudaFreeHost(copy_dst));
        CK(cudaFreeHost((void *)host));
        CK(cudaStreamDestroy(stream)); CK(cudaStreamDestroy(companion));
        CK(cudaStreamDestroy(copy_stream)); CK(cudaStreamDestroy(dependent));
        CK(cuCtxDestroy(ctx));
    }
};

int main(int argc, char **argv)
{
    CK(cuInit(0));
    CK(XHintSetScheduler(kSchedulerAppManaged, kPolicyUnknown));
    if (argc > 1 && std::strcmp(argv[1], "missing-tracker") == 0) {
        CK(cudaSetDevice(0));
        cudaStream_t stream;
        CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        HwQueueHandle hwq = 0;
        REQUIRE(CudaQueueCreate(&hwq, stream) != kXSchedSuccess && hwq == 0);
        CK(cudaStreamDestroy(stream));
        puts("PASS: unavailable CHANNEL rejected without registering a queue");
        return 0;
    }
    Task a, b;
    a.Create(2);
    b.Create(1);
    CudaChannelInfo info{};
    CK(CudaQueueGetChannelInfo(a.hwq, &info));
    REQUIRE(info.context == a.ctx && info.compute_group_scope &&
            info.independent_copies_may_run && info.preserves_inflight_work && !info.supports_drop_commands);
    REQUIRE(CudaQueueGetChannelInfo(a.hwq, nullptr) == kXSchedErrorInvalidValue);
    REQUIRE(HwQueueDestroy(a.hwq) == kXSchedErrorNotAllowed);
    a.Launch(); b.Launch();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while ((a.host[0] < 10 || a.host[1] < 10 || b.host[0] < 10) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(a.host[0] >= 10 && a.host[1] >= 10 && b.host[0] >= 10);
    CK(cuCtxSetCurrent(a.ctx));
    CK(XQueueSuspend(a.xq, kQueueSuspendFlagSyncHwQueue));
    unsigned p0 = a.host[0], p1 = a.host[1], pb = b.host[0];
    REQUIRE(XQueueResume(a.xq, kQueueResumeFlagDropCommands) == kXSchedErrorNotSupported);
    HwQueueHandle duplicate = 0;
    REQUIRE(CudaQueueCreate(&duplicate, a.companion) != kXSchedSuccess && duplicate == 0);
    REQUIRE(cuCtxDestroy(a.ctx) == CUDA_ERROR_NOT_READY);
    REQUIRE(cuDevicePrimaryCtxReset(0) == CUDA_ERROR_NOT_READY);
    REQUIRE(cudaEventQuery(a.kernel_done) == cudaErrorNotReady);
    REQUIRE(cudaEventQuery(a.dependent_done) == cudaErrorNotReady);
    for (unsigned i = 0; i < CopyWords; ++i) a.copy_src[i] ^= 0xa5a5a5a5;
    std::memset(a.copy_dst, 0, CopyWords * sizeof(unsigned));
    CK(cudaMemcpyAsync(a.copy_dev, a.copy_src, CopyWords * sizeof(unsigned), cudaMemcpyHostToDevice, a.copy_stream));
    CK(cudaMemcpyAsync(a.copy_dst, a.copy_dev, CopyWords * sizeof(unsigned), cudaMemcpyDeviceToHost, a.copy_stream));
    CK(cudaEventRecord(a.copy_done, a.copy_stream));
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (cudaEventQuery(a.copy_done) == cudaErrorNotReady && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto copy_state = cudaEventQuery(a.copy_done);
    REQUIRE(copy_state == cudaSuccess || copy_state == cudaErrorNotReady);
    if (copy_state == cudaSuccess)
        REQUIRE(std::memcmp(a.copy_src, a.copy_dst, CopyWords * sizeof(unsigned)) == 0);
    printf("Independent copy while compute paused: %s (forward progress is not guaranteed)\n",
           copy_state == cudaSuccess ? "completed" : "pending");
    std::atomic<bool> sync_started{false}, sync_done{false};
    std::thread waiter([&] {
        CK(cuCtxSetCurrent(a.ctx));
        sync_started.store(true);
        CK(HwQueueSynchronize(a.hwq));
        sync_done.store(true);
    });
    while (!sync_started.load()) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE(!sync_done.load());
    REQUIRE(a.host[0] == p0 && a.host[1] == p1 && b.host[0] > pb && b.host[0] < Limit);
    REQUIRE(cudaEventQuery(a.dependent_done) == cudaErrorNotReady);
    a.host[2] = 1;
    b.host[2] = 1;
    // Destroying a paused XQueue must resume and drain retained work.
    CK(XQueueDestroy(a.xq));
    waiter.join();
    REQUIRE(sync_done.load());
    CK(HwQueueDestroy(a.hwq));
    // Reuse the same context and stream after releasing ownership.
    CK(CudaQueueCreate(&a.hwq, a.stream));
    CK(XQueueCreate(&a.xq, a.hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone));
    auto toggle = [&] {
        for (int i = 0; i < 10; ++i) {
            CK(XQueueSuspend(a.xq, kQueueSuspendFlagSyncHwQueue));
            CK(XQueueResume(a.xq, kQueueResumeFlagNone));
        }
    };
    std::thread t0(toggle), t1(toggle);
    t0.join(); t1.join();
    CK(XQueueResume(a.xq, kQueueResumeFlagNone));
    a.CheckAndDestroy();
    b.CheckAndDestroy();
    puts("PASS: context isolation, copy/event semantics, sync, rejection, paused destruction, reuse, concurrent control");
}
