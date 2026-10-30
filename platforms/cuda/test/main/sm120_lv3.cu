#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cuda_runtime.h>
#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

#define CUDA_CHECK(call) do { auto err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); std::exit(1); } } while (0)
#define XS_CHECK(call) do { auto err = (call); if (err != kXSchedSuccess) { \
    fprintf(stderr, "%s: %d\n", #call, int(err)); std::exit(1); } } while (0)
#define REQUIRE(expr) do { if (!(expr)) { fprintf(stderr, "FAIL: %s\n", #expr); std::exit(1); } } while (0)

constexpr unsigned Iterations = 8000;
constexpr unsigned Threads = 128;
constexpr unsigned Blocks = 8;
constexpr unsigned Elements = Blocks * Threads;

__device__ unsigned BlockIndex()
{
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ unsigned ThreadIndex()
{
    return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}
__global__ void Work(volatile unsigned *progress, unsigned *counts, unsigned long long *result)
{
    unsigned block = BlockIndex(), thread = ThreadIndex();
    unsigned long long sum = 0;
    for (unsigned i = 0; i < Iterations; ++i) {
        auto start = clock64();
        while (clock64() - start < 500000ULL) {}
        sum += (unsigned long long)(i + 1) * (thread + 1);
        if (thread == 0) {
            atomicAdd(counts + block, 1u);
            if (block == 0) *progress = i + 1;
            __threadfence_system();
        }
    }
    result[block * Threads + thread] = sum;
}

// This queued kernel also checks ordering: it must see Work's final output.
__global__ void Tail(unsigned *counts, unsigned long long *result)
{
    unsigned block = BlockIndex(), thread = ThreadIndex();
    result[block * Threads + thread] += 17;
    if (thread == 0) atomicAdd(counts + block, 1u);
}

int main(int argc, char **argv)
{
    setbuf(stdout, nullptr);
    const char *impl = std::getenv("XSCHED_CUDA_LV3_IMPL");
    REQUIRE(impl && std::strcmp(impl, "CHANNEL") == 0);
    XS_CHECK(XHintSetScheduler(kSchedulerAppManaged, kPolicyUnknown));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    REQUIRE(prop.major == 12 && prop.minor == 0);
    volatile unsigned *progress;
    unsigned *device_progress, *counts;
    unsigned long long *result, *copied;
    CUDA_CHECK(cudaHostAlloc((void **)&progress, 2 * sizeof(unsigned), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&device_progress, (void *)progress, 0));
    CUDA_CHECK(cudaMalloc(&counts, 2 * Blocks * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&result, 2 * Elements * sizeof(*result)));
    CUDA_CHECK(cudaMallocHost(&copied, 2 * Elements * sizeof(*copied)));
    // Lazy loading of Tail during Work can serialize their first submissions.
    // Load both functions before requiring two simultaneously active streams.
    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, Work));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, Tail));

    // Recreating the managed queue checks that ownership is released correctly.
    for (int lifetime = 0; lifetime < 2; ++lifetime) {
        cudaStream_t stream, companion;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&companion, cudaStreamNonBlocking));
        HwQueueHandle hwq;
        XQueueHandle xq;
        XS_CHECK(CudaQueueCreate(&hwq, stream));
        if (argc > 1 && std::strcmp(argv[1], "duplicate") == 0) {
            HwQueueHandle duplicate = 0;
            REQUIRE(CudaQueueCreate(&duplicate, companion) != kXSchedSuccess && duplicate == 0);
            XS_CHECK(HwQueueDestroy(hwq));
            CUDA_CHECK(cudaStreamDestroy(stream));
            CUDA_CHECK(cudaStreamDestroy(companion));
            CUDA_CHECK(cudaFree(counts));
            CUDA_CHECK(cudaFree(result));
            CUDA_CHECK(cudaFreeHost(copied));
            CUDA_CHECK(cudaFreeHost((void *)progress));
            puts("PASS: duplicate context owner rejected without abort");
            return 0;
        }
        XS_CHECK(XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone));
        XS_CHECK(XQueueSetLaunchConfig(xq, 8, 4));
        CUDA_CHECK(cudaMemset(counts, 0, 2 * Blocks * sizeof(unsigned)));
        CUDA_CHECK(cudaDeviceSynchronize());
        progress[0] = 0;
        progress[1] = 0;
        const dim3 grid(2, 2, 2), block(8, 4, 4);
        Work<<<grid, block, 0, stream>>>(device_progress, counts, result);
        CUDA_CHECK(cudaGetLastError());
        Work<<<grid, block, 0, companion>>>(device_progress + 1, counts + Blocks, result + Elements);
        CUDA_CHECK(cudaGetLastError());
        Tail<<<grid, block, 0, stream>>>(counts, result);
        Tail<<<grid, block, 0, companion>>>(counts + Blocks, result + Elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(copied, result, Elements * sizeof(*result), cudaMemcpyDeviceToHost, stream));
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while ((progress[0] < 20 || progress[1] < 20) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        REQUIRE(progress[0] >= 20 && progress[1] >= 20);
        int64_t commands = 0;
        XS_CHECK(XQueueProfileHwCommandCount(xq, &commands));
        REQUIRE(commands >= 2); // Ensure the real shim/XQueue submission path is used.
        printf("managed_commands=%lld\n", (long long)commands);
        for (int round = 0; round < 3; ++round) {
            auto start = std::chrono::steady_clock::now();
            XS_CHECK(XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue));
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
            unsigned p0 = progress[0], p1 = progress[1];
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            unsigned q0 = progress[0], q1 = progress[1];
            bool frozen = q0 == p0 && q1 == p1 && p0 < Iterations && p1 < Iterations;
            XS_CHECK(XQueueResume(xq, kQueueResumeFlagNone));
            if (!frozen)
                fprintf(stderr, "pause observation before=(%u,%u) after=(%u,%u) suspend_us=%lld\n",
                        p0, p1, q0, q1, (long long)us);
            REQUIRE(frozen);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            REQUIRE(progress[0] > p0 && progress[1] > p1);
            printf("PASS lifetime=%d round=%d frozen=(%u,%u) suspend_us=%lld\n",
                   lifetime, round, p0, p1, (long long)us);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamSynchronize(companion));
        CUDA_CHECK(cudaMemcpy(copied + Elements, result + Elements, Elements * sizeof(*result), cudaMemcpyDeviceToHost));
        unsigned host_counts[2 * Blocks];
        CUDA_CHECK(cudaMemcpy(host_counts, counts, sizeof(host_counts), cudaMemcpyDeviceToHost));
        for (unsigned count : host_counts) REQUIRE(count == Iterations + 1);
        for (unsigned i = 0; i < Elements * 2; ++i)
            REQUIRE(copied[i] == (unsigned long long)Iterations * (Iterations + 1) / 2 * (i % Threads + 1) + 17);
        XS_CHECK(XQueueSetPreemptLevel(xq, kPreemptLevelBlock));
        XS_CHECK(XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue));
        XS_CHECK(XQueueResume(xq, kQueueResumeFlagNone));
        XS_CHECK(XQueueDestroy(xq));
        XS_CHECK(HwQueueDestroy(hwq));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaStreamDestroy(companion));
    }
    CUDA_CHECK(cudaFree(counts));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFreeHost(copied));
    CUDA_CHECK(cudaFreeHost((void *)progress));
    puts("PASS: CHANNEL Level 3, concurrent streams, 3D grid/block, queued kernels, exact results, Level 1 and queue recreation");
}
