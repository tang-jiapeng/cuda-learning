#include "cuda_utils.cuh"
#include "elementwise.h"

// 朴素实现：每个 thread 负责一个元素的加法运算
__global__ void elementwise_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void elementwise_add_gsl(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

void host_add_fp32(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

void kernel_add_fp32(int whichKernel, int blockSize, int gridSize, float* d_A, float* d_B,
                     float* d_C, int N) {
    void (*kernel)(const float*, const float*, float*, int);
    const char* kernelName = "";
    switch (whichKernel) {
        case 0:
            kernel = elementwise_add;
            kernelName = "elementwise_add";
            break;
        case 1:
            break;
        case 2:
            break;
        default:
            break;
    }
    printf("kernel: [%s], grid: [%d], block: [%d], N: [%d] \n", kernelName, gridSize,
           blockSize, N);
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
}