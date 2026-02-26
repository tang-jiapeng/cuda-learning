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

// 每个 Block 负责计算 C 中 (Bm, Bn) 大小的分块 tileC
template <int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
          int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void blockTileGEMM(float* A, float* B, float* C, const int M, const int K,
                              const int N) {
    __shared__ float As[Bm][Bk];  // 存储 tileA
    __shared__ float Bs[Bk][Bn];  // 存储 tileB

    // 计算 block 负责的 tileC 左上角元素的行列坐标(r0,c0)
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    // 当前 thread 的编号（默认为一维 block 配置）
    int tid = threadIdx.x;

    /*------ tileA ------*/
    // 写入 A tile 时，block 中 thread 排布尺寸为(A_BLOCK_X, blockSize / A_BLOCK_X)
    constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;  // (8,32)

    // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
    int A_THREAD_X = tid % A_BLOCK_X;
    int A_THREAD_Y = tid / A_BLOCK_Y;

    /*------ tileB ------*/
    // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X)
    constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;  // (32, 8)

    // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
    int B_THREAD_X = tid % B_BLOCK_X;
    int B_THREAD_Y = tid / B_BLOCK_Y;

    /*------ tileC ------*/
    constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

    // 对于 tid 号线程，其位于 blockC 中的行列坐标为 (tid / C_BLOCK_X, tid % C_BLOCK_X)
    int C_THREAD_Y = tid / C_BLOCK_X;
    int C_THREAD_X = tid % C_BLOCK_X;

    // 每个 thread 负责 Tm * Tn 个元素计算
    constexpr int Tm = Bm / C_BLOCK_Y;
    constexpr int Tn = Bn / C_BLOCK_X;
    float Ct[Tm][Tn] = {0.0};

    // K- Loop
    for (int k = 0; k < K; k += Bk) {
        /* ------ 读取 global memory，存入 shared memory ------ */
        // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
        for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
            int r = r0 + i;
#pragma unroll
            for (int j = A_THREAD_X; j < Bn; j += A_BLOCK_X) {
                int c = c0 + j;
                As[i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
            }
        }

        // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
        for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
            int r = r0 + i;
#pragma unroll
            for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
                int c = c0 + j;
                Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
            }
        }

        __syncthreads();

/* ------ 计算 tileA * tileB ------ */
// 先循环 k 维度，按向量外积的方式计算
#pragma unroll
        for (int p = 0; p < Bk; ++p) {
#pragma unroll
            // 使用跨步循环，行方向的 stride 为 C_BLOCK_Y, 列方向的 stride 为 C_BLOCK_X
            for (int i = 0; i < Tm; ++i) {
                int r = C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
                for (int j = 0; j < Tn; ++j) {
                    int c = C_THREAD_X + j * C_BLOCK_X;
                    Ct[r][c] += As[r][p] * Bs[p][c];
                }
            }
        }

        __syncthreads();
    }

    // 将 Ct 写入 C
#pragma unroll
    for (int i = 0; i < Tm; ++i) {
        int r = r0 + C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
        for (int j = 0; j < Tn; ++j) {
            int c = c0 + C_THREAD_X + j * C_BLOCK_X;
            if (r < M && c < N) C[r * N + c] = Ct[i][j];
        }
    }
}

std::string get_kernel_name(int kernel_num) {
    switch (kernel_num) {
        case 0:
            return "naiveGEMM";
        default:
            return "Unknown";
    }
}

void host_gemm(float* A, float* B, float* out, const int M, const int K, const int N) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float val = 0.0f;
            for (int k = 0; k < K; ++k) {
                val += A[r * K + k] * B[k * N + c];
            }
            out[r * N + c] = val;
        }
    }
}

void lauch_gemm_kernel(int whichKernel, float* A, float* B, float* out, const int M,
                       const int K, const int N) {
    const int Bm = 128;
    const int Bn = 128;

    switch (whichKernel) {
        case 0: {
            dim3 block(16, 16);
            dim3 grid(CEIL(N, block.x), CEIL(M, block.y));
            naiveGEMM<<<grid, block>>>(A, B, out, M, K, N);
            break;
        }
        case 1: {
            dim3 block = 256;
            dim3 grid(CEIL(N, Bn), CEIL(M, Bm));
            blockTileGEMM<<<grid, block>>>(A, B, out, M, K, N);
        }
        default:
            break;
    }
}