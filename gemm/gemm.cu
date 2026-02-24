#include "gemm.h"

#define WARP_SIZE 32
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// 每个 thread 负责计算 C 中的一个元素
__global__ void naiveGEMM(float* A, float* B, float* C, const int M, const int K,
                          const int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= M || c >= N) return;

    float value = 0.f;
    for (int k = 0; k < K; ++k) {
        value += A[r * K + k] * B[k * N + c];
    }
    C[r * N + c] = value;
}

