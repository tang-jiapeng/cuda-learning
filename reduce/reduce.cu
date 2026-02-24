#include <algorithm>
#include "reduce.h"

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// version 0: 简单实现，存在线程束分化问题
__global__ void reduce0(float* d_A, const int N) {
    // 申请 shared memory，用于存放 block 负责的数据
    extern __shared__ float data[];

    // 读取数据到 shared memory
    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    data[tid] = index < N ? d_A[index] : 0.f;

    __syncthreads();

    // iter0 [s = 1] (t0, t2, t4, ...): t0 -> (0, 1) | t2 -> (2, 3) | t4 -> (4, 5) ...
    // iter1 [s = 2] (t0, t4, t8, ...): t0 -> (0, 2) | t4 -> (4, 6) | t8 -> (8, 10) ...
    for (int s = 1; s < blockDim.x; s <<= 1) {
        // 负责执行运算的 threadIdx 为 0, 2s, 4s, 8s, ...
        if ((tid % (s * 2)) == 0) {
            data[tid] += data[tid + s];
        }
    }

    __syncthreads();

    // block 负责数据的求和结果存储在 data[0]，由 0 号 thread 写入 d_A[blockIdx.x] 中
    if (tid == 0) {
        d_A[blockIdx.x] = data[0];
    }
}

// version 0.5: 优化取余运算
__global__ void reduce0_5(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    data[tid] = index < N ? d_A[index] : 0.f;

    __syncthreads();

    for (int s = 1; s < blockDim.x; s <<= 1) {
        // a & (b-1) 代替取余运算
        if ((tid & (s * 2 - 1)) == 0) {
            data[tid] += data[tid + s];
        }
    }

    __syncthreads();

    if (tid == 0) {
        d_A[blockIdx.x] = data[0];
    }
}

// version1: 使用连续 thread 负责计算，解决线程束分化，存在 bank conflicts
__global__ void reduce1(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    data[tid] = i < N ? d_A[i] : 0.f;

    __syncthreads();

    // iter0 [s = 1] (t0 ~ tN/2): t0 -> (0, 1) | t1 -> (2, 3) | t2 -> (4, 5) ...
    // iter1 [s = 2] (t0 ~ tN/4): t0 -> (0, 2) | t1 -> (4, 6) | t2 -> (8, 10) ...
    for (int s = 1; s < blockDim.x; s <<= 1) {
        int index = tid * s * 2;
        // 使用连续 thread 执行运算，需要设置 blockDim.x 为偶数，避免越界
        if (index < blockDim.x) {
            data[index] += data[index + s];
        }
    }

    __syncthreads();

    if (tid == 0) {
        d_A[blockIdx.x] = data[0];
    }
}

// version2: 步长从大到小变化，解决 bank conflicts
__global__ void reduce2(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    data[tid] = i < N ? d_A[i] : 0.f;

    __syncthreads();

    // iter0 [s = N/2] (t0 ~ tN/2-1):
    // t0 -> (0, N/2) | t1 -> (1, N/2 + 1) | t2 -> (2, N/2 + 2) ...
    // iter1 [s = N/4] (t0 ~ tN/4-1):
    // t0 -> (0, N/4) | t1 -> (1, N/4 + 1) | t2 ->(2, N/4 + 2) ...
    // stride 从大到小变化，避免 bank conflicts
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            data[tid] += data[tid + s];
        }
    }

    __syncthreads();

    if (tid == 0) {
        d_A[blockIdx.x] = data[0];
    }
}

// version3: 读取数据到 shared memory 时，进行一次加法运算
__global__ void reduce3(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    // 每个 block 负责 2 * blockDim.x 个元素
    int i = blockDim.x * blockIdx.x + threadIdx.x * 2;

    // 将第 i 和 i + blockDim.x 个元素求和，结果写入到 shared memory 中（需要避免越界）
    float sum = i < N ? d_A[i] : 0.f;
    if (i + blockDim.x < N) {
        sum += d_A[i + blockDim.x];
    }
    data[tid] = sum;

    __syncthreads();

    // stride 从大到小变化，避免 bank conflicts
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            data[tid] += data[tid + s];
        }
    }

    __syncthreads();

    if (tid == 0) {
        d_A[blockIdx.x] = data[0];
    }
}

// 使用 __shfl_xor_sync 实现 WarpReduce
template <int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warpReduce(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// version4: 最后 32 个数使用 warp shuffle 完成求和
__global__ void reduce4(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    // 每个 block 负责 2 * blockDim.x 个元素
    int i = blockDim.x * blockIdx.x + threadIdx.x * 2;

    // 将第 i 和 i + blockDim.x 个元素求和，结果写入到 shared memory 中（需要避免越界）
    float sum = i < N ? d_A[i] : 0.f;
    if (i + blockDim.x < N) {
        sum += d_A[i + blockDim.x];
    }
    data[tid] = sum;

    __syncthreads();

    // stride 从大到小变化，避免 bank conflicts，循环条件变为 s >= 32
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) {
            data[tid] = sum = sum + data[tid + s];
        }
    }

    __syncthreads();

    // WarpReduce
    if (tid < 32) {
        sum = warpReduce<WARP_SIZE>(sum);
    }

    if (tid == 0) {
        d_A[blockIdx.x] = data[0];
    }
}

// version5: 展开循环
// 将 blockSize 作为模板参数，可以在编译期确定其数值，进而优化 if 分支
template <int blockSize>
__global__ void reduce5(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x * 2;

    float sum = i < N ? d_A[i] : 0.f;
    if (i + blockDim.x < N) {
        sum += d_A[i + blockDim.x];
    }
    data[tid] = sum;

    __syncthreads();

    // 依据 blockSize 大小，展开循环
    if (blockSize >= 1024 && tid < 512) data[tid] = sum = sum + data[tid + 512];
    __syncthreads();

    if (blockSize >= 512 && tid < 256) data[tid] = sum = sum + data[tid + 256];
    __syncthreads();

    if (blockSize >= 256 && tid < 128) data[tid] = sum = sum + data[tid + 128];
    __syncthreads();

    if (blockSize >= 128 && tid < 64) data[tid] = sum = sum + data[tid + 64];
    __syncthreads();

    if (blockSize >= 64 && tid < 32) data[tid] = sum = sum + data[tid + 32];
    __syncthreads();

    if (tid < 32) {
        sum = warpReduce<WARP_SIZE>(sum);
    }

    if (tid == 0) {
        d_A[blockIdx.x] = sum;
    }
}

// version6:读取数据到 shared memory 时，进行多次加法运算
template <int blockSize>
__global__ void reduce6(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // 网格跨步循环，对多个元素求和
    float sum = 0.f;
    for (int index = i; index < N; index += blockSize * gridDim.x) {
        sum += d_A[index];
    }
    data[tid] = sum;

    __syncthreads();

    // 依据 blockSize 大小，展开循环
    if (blockSize >= 1024 && tid < 512) data[tid] = sum = sum + data[tid + 512];
    __syncthreads();

    if (blockSize >= 512 && tid < 256) data[tid] = sum = sum + data[tid + 256];
    __syncthreads();

    if (blockSize >= 256 && tid < 128) data[tid] = sum = sum + data[tid + 128];
    __syncthreads();

    if (blockSize >= 128 && tid < 64) data[tid] = sum = sum + data[tid + 64];
    __syncthreads();

    if (blockSize >= 64 && tid < 32) data[tid] = sum = sum + data[tid + 32];
    __syncthreads();

    if (tid < 32) {
        sum = warpReduce<WARP_SIZE>(sum);
    }

    if (tid == 0) {
        d_A[blockIdx.x] = sum;
    }
}

// version6_vec4: 使用 float4 读取数据并求和，然后写入到 shared memory 中
template <int blockSize>
__global__ void reduce6_vec4(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    int i = 4 * (blockSize * blockIdx.x + threadIdx.x);  // 注意索引要乘以 4

    // 向量化访存
    float sum = 0.f;
    if (i < N - 4) {
        float4 reg = FLOAT4(d_A[i]);
        sum = reg.x + reg.y + reg.z + reg.w;
    } else {  // 不足 4 个元素时进行特殊处理
        for (int j = i; j < N; ++j) {
            sum += d_A[j];
        }
    }
    data[tid] = sum;

    __syncthreads();

    // 依据 blockSize 大小，展开循环
    if (blockSize >= 1024 && tid < 512) data[tid] = sum = sum + data[tid + 512];
    __syncthreads();

    if (blockSize >= 512 && tid < 256) data[tid] = sum = sum + data[tid + 256];
    __syncthreads();

    if (blockSize >= 256 && tid < 128) data[tid] = sum = sum + data[tid + 128];
    __syncthreads();

    if (blockSize >= 128 && tid < 64) data[tid] = sum = sum + data[tid + 64];
    __syncthreads();

    if (blockSize >= 64 && tid < 32) data[tid] = sum = sum + data[tid + 32];
    __syncthreads();

    if (tid < 32) {
        sum = warpReduce<WARP_SIZE>(sum);
    }

    if (tid == 0) {
        d_A[blockIdx.x] = sum;
    }
}

// version7: 在最开始写入 shared memory 之前，进行一次 warp reduction
template <int blockSize>
__global__ void reduce7(float* d_A, const int N) {
    extern __shared__ float data[];

    int tid = threadIdx.x;
    int i = blockSize * blockIdx.x + threadIdx.x;

    // 网格跨步循环，对多个元素求和
    float sum = 0.f;
    for (int index = i; index < N; index += blockSize * gridDim.x) {
        sum += d_A[index];
    }

    // 每个 warp 都执行 WarpReduce
    sum = warpReduce<WARP_SIZE>(sum);
    // WarpReduce 结果按照 warp ID 写入 shared memory
    if ((tid & (WARP_SIZE - 1)) == 0) {
        data[tid / WARP_SIZE] = sum;
    }

    __syncthreads();

    // 只在 warp 0 执行 WarpReduce
    constexpr int NUM_WARPS = CEIL(blockSize, WARP_SIZE);
    if (tid < 32) {
        // 只保留 NUM_WARPS 个有效数据
        sum = tid < NUM_WARPS ? data[tid] : 0.f;
        sum = warpReduce<NUM_WARPS>(sum);
    }

    if (tid == 0) {
        d_A[blockIdx.x] = sum;
    }
}

float pairwise_sum(float* A, int start, int stop) {
    if (stop == start) {
        return A[start];
    }

    int mid = (start + stop) >> 1;
    return pairwise_sum(A, start, mid) + pairwise_sum(A, mid + 1, stop);
}

float host_reduce_sum(float* A, const int N) {
    return pairwise_sum(A, 0, N - 1);
}

// ---------------------------------------------------------------------------
// Kernel registry
// ---------------------------------------------------------------------------

std::string get_kernel_name(int kernel_num) {
    switch (kernel_num) {
        case 0:
            return "reduce0  (naive, warp diverge)";
        case 1:
            return "reduce0_5 (bitwise mod)";
        case 2:
            return "reduce1  (no diverge, bank conflict)";
        case 3:
            return "reduce2  (stride-down, no bank conflict)";
        case 4:
            return "reduce3  (first-add on load)";
        case 5:
            return "reduce4  (warp shuffle last 32)";
        case 6:
            return "reduce5<256> (loop unroll)";
        case 7:
            return "reduce6<256> (grid-stride)";
        case 8:
            return "reduce6_vec4<256> (grid-stride + vec4)";
        case 9:
            return "reduce7<256> (warp-reduce-first)";
        default:
            return "Unknown";
    }
}

// ---------------------------------------------------------------------------
// Multi-stage reduction launcher
//
// All kernels write one partial sum per block back to d_A[blockIdx.x], so we
// iterate until a single element remains.  d_A is modified in-place.
// ---------------------------------------------------------------------------

static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

void launch_reduce_sum_kernel(int whichKernel, float* d_A, const int N) {
    constexpr int kBlockSize = 256;
    constexpr int kMaxGridSz = 1024;  // cap for grid-stride kernels (pass 1)

    int remaining = N;

    while (remaining > 1) {
        int grid = 0;
        int smem = kBlockSize * static_cast<int>(sizeof(float));

        switch (whichKernel) {
            // ---- 1 element per thread ----
            case 0:
                grid = ceil_div(remaining, kBlockSize);
                reduce0<<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            case 1:
                grid = ceil_div(remaining, kBlockSize);
                reduce0_5<<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            case 2:
                grid = ceil_div(remaining, kBlockSize);
                reduce1<<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            case 3:
                grid = ceil_div(remaining, kBlockSize);
                reduce2<<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            // ---- 2 elements per thread on load ----
            case 4:
                grid = ceil_div(remaining, kBlockSize * 2);
                reduce3<<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            case 5:
                grid = ceil_div(remaining, kBlockSize * 2);
                reduce4<<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            case 6:
                grid = ceil_div(remaining, kBlockSize * 2);
                reduce5<kBlockSize><<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            // ---- grid-stride loop: cap grid to kMaxGridSz for 1st pass ----
            case 7:
                grid = std::min(ceil_div(remaining, kBlockSize), kMaxGridSz);
                reduce6<kBlockSize><<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            // ---- vec4 grid-stride: 4 elements per thread ----
            case 8:
                grid = ceil_div(remaining, kBlockSize * 4);
                if (grid == 0) grid = 1;
                reduce6_vec4<kBlockSize><<<grid, kBlockSize, smem>>>(d_A, remaining);
                break;
            // ---- warp-first grid-stride ----
            case 9: {
                grid = std::min(ceil_div(remaining, kBlockSize), kMaxGridSz);
                // smem only needs NUM_WARPS floats; use full block allocation for safety
                constexpr int kNumWarps = CEIL(kBlockSize, WARP_SIZE);
                reduce7<kBlockSize>
                    <<<grid, kBlockSize, kNumWarps * sizeof(float)>>>(d_A, remaining);
                break;
            }
            default:
                return;
        }
        remaining = grid;
    }
}