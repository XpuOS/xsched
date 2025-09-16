#include <chrono>
#include <thread>
#include <random>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"

#define VECTOR_SIZE (1 << 25) // 32MB
#define N 100    // Number of vector additions per task
#define M 10000  // Number of tasks, (almost) never stops

// 1 is for low priority, 2 is for high priority
cudaStream_t stream_1, stream_2;
HwQueueHandle hwq_1, hwq_2;
XQueueHandle xq_1, xq_2;

__global__ void vector_add(const float* A, const float* B, float* C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    C[i] = A[i] + B[i];
}

void task(float *d_A, float *d_B, float *d_C, cudaStream_t stream)
{
    // Launch kernel N times
    int block_size = 256;
    int grid_size = (VECTOR_SIZE + block_size - 1) / block_size;
    
    for (int i = 0; i < N; ++i) {
        vector_add<<<grid_size, block_size, 0, stream>>>(d_A, d_B, d_C, VECTOR_SIZE);
    }
    cudaStreamSynchronize(stream);
}

void run(bool is_high_priority)
{
    cudaStream_t stream = is_high_priority ? stream_2 : stream_1;

    // Global memory pointers
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size = VECTOR_SIZE * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Run tasks
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(30, 50);

    for (int i = 0; i < M; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        // If this is the high-priority task,
        // suspend the low-priority task when the high-priority task starts.
        if (is_high_priority) XQueueSuspend(xq_1, 0);
        task(d_A, d_B, d_C, stream);
        // If this is the high-priority task,
        // resume the low-priority task when the high-priority task finishes.
        if (is_high_priority) XQueueResume(xq_1, 0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("%s prio Task %d completed in %ld ms\n", is_high_priority ? "high" : "low ", i, duration.count());

        // Sleep for random interval between tasks
        std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main()
{
    cudaStreamCreate(&stream_1);
    CudaQueueCreate(&hwq_1, stream_1);
    XQueueCreate(&xq_1, hwq_1, kPreemptLevelBlock, kQueueCreateFlagNone);
    XQueueSetLaunchConfig(xq_1, 8, 4);

    cudaStreamCreate(&stream_2);
    CudaQueueCreate(&hwq_2, stream_2);
    XQueueCreate(&xq_2, hwq_2, kPreemptLevelBlock, kQueueCreateFlagNone);
    XQueueSetLaunchConfig(xq_2, 8, 4);

    // run two tasks within one process
    std::thread thread_lp(run, false);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::thread thread_hp(run, true);

    thread_lp.join();
    thread_hp.join();

    return 0;
}
