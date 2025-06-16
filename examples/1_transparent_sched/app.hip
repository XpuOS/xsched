#include <chrono>
#include <thread>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define VECTOR_SIZE (1 << 25) // 32MB
#define N 100    // Number of vector additions per task
#define M 10000  // Number of tasks, (almost) never stops

// Global memory pointers
float *h_A, *h_B, *h_C;
float *d_A, *d_B, *d_C;

hipStream_t stream;

__global__ void vector_add(const float* A, const float* B, float* C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    C[i] = A[i] + B[i];
}

void prepare()
{
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
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    // Copy vectors to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    hipStreamCreate(&stream);
}

void run_task()
{
    // Launch kernel N times
    int block_size = 256;
    int grid_size = (VECTOR_SIZE + block_size - 1) / block_size;
    
    for (int i = 0; i < N; ++i) {
        vector_add<<<grid_size, block_size, 0, stream>>>(d_A, d_B, d_C, VECTOR_SIZE);
    }
    hipStreamSynchronize(stream);
}

void cleanup()
{
    // Free memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(30, 50);

    prepare();

    // Run tasks
    for (int i = 0; i < M; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        run_task();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Task %d completed in %ld ms\n", i, duration.count());

        // Sleep for random interval between tasks
        std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
    }

    cleanup();
    return 0;
}
