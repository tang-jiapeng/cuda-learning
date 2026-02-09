#include "cuda_utils.cuh"
#include "elementwise.h"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// 朴素实现：每个 thread 负责一个元素的加法运算
__global__ void elementwise_add(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// grid_size = 4 , block_size = 256, 1024 * 1024 * 16
// Grid-Stride Loops 网格跨步循环：每个 thread 可以负责多个元素运算，间隔为总的 thread
// 数量 优点：wrap 内 thread 的内存访问连续；可以使用任意数量的 thread 完成 N 个元素运算
__global__ void elementwise_add_gsl(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nums_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += nums_threads) {
        C[i] = A[i] + B[i];
    }
}

// Vectorized Memory Access 向量化访存：使用一条 LD.E.128 指令代替四条 LD.E 指令，
// 优点：可以一次读取 4 个 float 数据，降低延迟，提高带宽
__global__ void elementwise_add_vec4(float* A, float* B, float* C, const int N) {
    int index = 4 * (threadIdx.x + blockIdx.x * blockDim.x);

    if (index > N) {
        return;
    }

    if (index <= N - 4) {
        float4 a = FLOAT4(A[index]);
        float4 b = FLOAT4(B[index]);
        float4 c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        FLOAT4(C[index]) = c;
    } else {
#pragma unroll
        for (int i = index; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

// 向量化访存 + 网格跨步循环
__global__ void elementwise_add_vec4_gsl(float* A, float* B, float* C, int N) {
    int tid = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    for (int i = tid; i < N; i += 4 * blockDim.x * gridDim.x) {
        if (i < N - 4) {
            float4 a_vec = FLOAT4(A[i]);
            float4 b_vec = FLOAT4(B[i]);
            float4 c_vec;

            c_vec.x = a_vec.x + b_vec.x;
            c_vec.y = a_vec.y + b_vec.y;
            c_vec.z = a_vec.z + b_vec.z;
            c_vec.w = a_vec.w + b_vec.w;
            FLOAT4(C[i]) = c_vec;
        } else {
#pragma unroll
            for (int j = i; j < N; ++j) {
                C[j] = A[j] + B[j];
            }
        }
    }
}

void host_add_fp32(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

std::string get_kernel_name(int kernel_num) {
    switch (kernel_num) {
        case 0:
            return "Naive (One-to-One)";
        case 1:
            return "Grid-Stride Loop";
        case 2:
            return "Vectorized (float4)";
        case 3:
            return "Vectorized + GSL";
        default:
            return "Unknown";
    }
}

void launch_elementwise_add_kernel(int whichKernel, int blockSize, int userGridSize,
                                   float* d_A, float* d_B, float* d_C, int N) {
    int gridSize = userGridSize;

    // 如果没有指定 Grid Size (即为0)，则根据 Kernel 类型自动计算
    if (gridSize <= 0) {
        if (whichKernel == 2 || whichKernel == 3) {
            gridSize = (N + blockSize * 4 - 1) / (blockSize * 4);
        } else {
            gridSize = (N + blockSize - 1) / blockSize;
        }
    }

    switch (whichKernel) {
        case 0:
            elementwise_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
            break;
        case 1:
            elementwise_add_gsl<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
            break;
        case 2:
            elementwise_add_vec4<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
            break;
        case 3:
            elementwise_add_vec4_gsl<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
            break;
    }
}