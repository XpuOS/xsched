#include <chrono>
#include <thread>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <windows.h>
#include <cuda_runtime.h>

#define VECTOR_SIZE (1 << 25) // 32MB
#define N 100    // Number of vector additions per task
#define M 10000  // Number of tasks, (almost) never stops
#define EPSILON 1e-5f  // Error tolerance for floating-point comparison

__global__ void vector_add(const float* A, const float* B, float* C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    C[i] = A[i] + B[i];
}

bool verify_result(const float* h_A, const float* h_B, const float* h_C, int n) {
    for (int i = 0; i < n; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > EPSILON) {
            printf("Verification failed at element %d: expected %f, got %f\n", 
                   i, expected, h_C[i]);
            return false;
        }
    }
    return true;
}

void task(float *d_A, float *d_B, float *d_C, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (VECTOR_SIZE + block_size - 1) / block_size;

    // Launch kernel N times
    for (int i = 0; i < N; ++i) {
        vector_add<<<grid_size, block_size, 0, stream>>>(d_A, d_B, d_C, VECTOR_SIZE);
    }
    cudaStreamSynchronize(stream);
}

void run(cudaStream_t stream)
{
    size_t size = VECTOR_SIZE * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

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

    // Run tasks
    for (int i = 0; i < M; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        task(d_A, d_B, d_C, stream);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Task %d completed in %lld ms\n", i, duration.count());
        
        // // Copy result back to host
        // cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        // // Verify the result
        // bool is_correct = verify_result(h_A, h_B, h_C, VECTOR_SIZE);
        // if (!is_correct) {
        //     printf("Task %d: Result verification failed!\n", i);
        // }
        // // Clear device memory for next iteration
        // cudaMemset(d_C, 0, size);
        // memset(h_C, 0, size);
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
    int thread_num = 1; // Number of threads to run

    std::vector<std::thread> threads;

    for (int i = 0; i < thread_num; ++i) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        std::thread thread(run, stream);
        threads.push_back(std::move(thread));
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}
