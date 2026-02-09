#include "transpose.h"

// A: MxN => B: NxM

// 0 <= ix < N, 0 <= iy < M
// 朴素实现 NaiveRow：每个 thread 负责一个元素的转置，按行读取，按列写入
__global__ void transposeNativeRow(float* A, float* B, const int M, const int N) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < N && iy < M) {
        B[ix * M + iy] = A[iy * N + ix];
    }
}

// 朴素实现 NaiveCol：每个 thread 负责一个元素的转置，按列读取，按行写入
__global__ void transposeNativeCol(float* A, float* B, const int M, const int N) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < M && iy < N) {
        B[iy * M + ix] = A[ix * N + iy];
    }
}

// 每个 block 负责一个分块 tile，每个 thread 负责多个元素
template <int Bm, int Bn>
__global__ void transposeColNelements(float* A, float* B, const int M, const int N) {
    // (r0, c0) 表示 tile 内左上角元素的坐标
    int r0 = blockIdx.x * Bm;
    int c0 = blockIdx.y * Bn;

    // 循环中的 (x, y) 表示 thread 负责的元素在 tile 内的坐标 (以左上角为 (0, 0))
    // thread 负责的元素的实际坐标为: (r0 + x, c0 + y)
#pragma unroll
    // 在 x 方向，每次跨度为 blockDim.x
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
        int r = r0 + x;
        if (r >= M) break;
#pragma unroll
        // 在 y 方向，每次跨度为 blockDim.y
        for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
            int c = c0 + y;
            if (c < N) {
                B[c * M + r] = A[r * N + c];
            }
        }
    }
}

template <int Bm, int Bn>
__global__ void transposeShared(float* A, float* B, const int M, const int N) {
    __shared__ float tile[Bm][Bn];

    /* -------- 读取阶段 -------- */
    // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    // thread y 方向负责：矩阵 A 的行，shared memory 的行
    // thread x 方向负责：矩阵 A 的列，shared memory 的列
    // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]
#pragma unroll
    // 在 y 方向，每次跨度为 blockDim.y
    for (int y = threadIdx.y; y < Bm; y += blockDim.y) {
        int r = r0 + y;
        if (r >= M) break;
#pragma unroll
        // 在 x 方向，每次跨度为 blockDim.x
        for (int x = threadIdx.x; x < Bn; x += blockDim.x) {
            int c = c0 + x;
            if (c < N) {
                tile[y][x] = A[r * N + c];  // 将 A[r0 + y, c0 + x] 写入 tile[y][x]
            }
        }
    }

    __syncthreads();  // 同步线程块

    /* -------- 写入阶段 -------- */
    // (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
    // thread y 方向负责：矩阵 B 的行，shared memory 的列
    // thread x 方向负责：矩阵 B 的列，shared memory 的行
    // shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
#pragma unroll
    // 在 y 方向，每次跨度为 blockDim.y
    for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
        int c = c0 + y;
        if (c >= N) break;
#pragma unroll
        // 在 x 方向，每次跨度为 blockDim.x
        for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
            int r = r0 + x;
            if (r < M) {
                B[c * M + r] = tile[x][y];  // 将 tile[x][y] 写入 B[c0 + y, r0 + x]
            }
        }
    }
}

template <int Bm, int Bn>
__global__ void transposeSharedPadding(float* A, float* B, const int M, const int N) {
    // 在 shared memory 中，每行多分配一个元素，避免 bank conflict
    __shared__ float tile[Bm][Bn + 1];

    /* -------- 读取阶段 -------- */
    // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    // thread y 方向负责：矩阵 A 的行，shared memory 的行
    // thread x 方向负责：矩阵 A 的列，shared memory 的列
    // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]
#pragma unroll
    // 在 y 方向，每次跨度为 blockDim.y
    for (int y = threadIdx.y; y < Bm; y += blockDim.y) {
        int r = r0 + y;
        if (r >= M) break;
#pragma unroll
        // 在 x 方向，每次跨度为 blockDim.x
        for (int x = threadIdx.x; x < Bn; x += blockDim.x) {
            int c = c0 + x;
            if (c < N) {
                tile[y][x] = A[r * N + c];  // 将 A[r0 + y, c0 + x] 写入 tile[y][x]
            }
        }
    }

    __syncthreads();  // 同步线程块

    /* -------- 写入阶段 -------- */
    // (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
    // thread y 方向负责：矩阵 B 的行，shared memory 的列
    // thread x 方向负责：矩阵 B 的列，shared memory 的行
    // shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
#pragma unroll
    // 在 y 方向，每次跨度为 blockDim.y
    for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
        int c = c0 + y;
        if (c >= N) break;
#pragma unroll
        // 在 x 方向，每次跨度为 blockDim.x
        for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
            int r = r0 + x;
            if (r < M) {
                B[c * M + r] = tile[x][y];  // 将 tile[x][y] 写入 B[c0 + y, r0 + x]
            }
        }
    }
}

template <int Bm, int Bn>
__global__ void transposeSharedSwizzling(float* A, float* B, const int M, const int N) {
    __shared__ float tile[Bm][Bn];

    /* -------- 读取阶段 -------- */
    // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    // thread y 方向负责：矩阵 A 的行，shared memory 的行
    // thread x 方向负责：矩阵 A 的列，shared memory 的列
    // shared memory 中的元素 tile[y][x ^ y] = A[r0 + y, c0 + x]
#pragma unroll
    // 在 y 方向，每次跨度为 blockDim.y
    for (int y = threadIdx.y; y < Bm; y += blockDim.y) {
        int r = r0 + y;
        if (r >= M) break;

#pragma unroll
        // 在 x 方向，每次跨度为 blockDim.x
        for (int x = threadIdx.x; x < Bn; x += blockDim.x) {
            int c = c0 + x;
            if (c < N) {
                // 将 A[r0 + y, c0 + x] 写入 tile[y][x ^ y]
                tile[y][x ^ y] = A[r * N + c];
            }
        }
    }

    __syncthreads();  // 同步线程块

    /* -------- 写入阶段 -------- */
    // (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
    // thread y 方向负责：矩阵 B 的行，shared memory 的列
    // thread x 方向负责：矩阵 B 的列，shared memory 的行
    // shared memory 中的元素 tile[x][x ^ y] = B[c0 + y, r0 + x]
#pragma unroll
    // 在 y 方向，每次跨度为 blockDim.y
    for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
        int c = c0 + y;
        if (c >= N) break;

#pragma unroll
        // 在 x 方向，每次跨度为 blockDim.x
        for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
            int r = r0 + x;
            if (r < M) {
                // 将 tile[x][x ^ y] 写入 B[c0 + y, r0 + x]
                B[c * M + r] = tile[x][x ^ y];
            }
        }
    }
}

__global__ void transposeSharedPaddingUnroll(float* A, float* B, const int M,
                                             const int N) {
    __shared__ float tile[32][33];  // padding

    /* -------- 读取阶段 -------- */
    // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
    int r0 = blockIdx.y * 32;
    int c0 = blockIdx.x * 32;

    // thread y 方向负责：矩阵 A 的行，shared memory 的行
    // thread x 方向负责：矩阵 A 的列，shared memory 的列
    // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]

    int y = threadIdx.y;
    int r = r0 + y;
    int c = c0 + threadIdx.x;
    if (c >= N) return;

    if (r < M) {
        tile[y][threadIdx.x] = A[r * N + c];
    }

    y += blockDim.y;
    r += blockDim.y;
    if (r < M) {
        tile[y][threadIdx.x] = A[r * N + c];
    }

    y += blockDim.y;
    r += blockDim.y;
    if (r < M) {
        tile[y][threadIdx.x] = A[r * N + c];
    }

    y += blockDim.y;
    r += blockDim.y;
    if (r < M) {
        tile[y][threadIdx.x] = A[r * N + c];
    }

    __syncthreads();  // 同步线程块

    /* -------- 写入阶段 -------- */
    // (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
    // thread y 方向负责：矩阵 B 的行，shared memory 的列
    // thread x 方向负责：矩阵 B 的列，shared memory 的行
    // shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
    r = r0 + threadIdx.x;
    if (r >= M) return;

    y = threadIdx.y;
    c = c0 + y;
    if (c < N) {
        B[c * M + r] = tile[threadIdx.x][y];
    }

    y += blockDim.y;
    c += blockDim.y;
    if (c < N) {
        B[c * M + r] = tile[threadIdx.x][y];
    }

    y += blockDim.y;
    c += blockDim.y;
    if (c < N) {
        B[c * M + r] = tile[threadIdx.x][y];
    }

    y += blockDim.y;
    c += blockDim.y;
    if (c < N) {
        B[c * M + r] = tile[threadIdx.x][y];
    }
}

__global__ void transposeSharedSwizzlingUnroll(float* A, float* B, const int M,
                                               const int N) {
    __shared__ float tile[32][32];

    /* -------- 读取阶段 -------- */
    // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
    int r0 = blockIdx.y * 32;
    int c0 = blockIdx.x * 32;

    // thread y 方向负责：矩阵 A 的行，shared memory 的行
    // thread x 方向负责：矩阵 A 的列，shared memory 的列
    // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]

    int y = threadIdx.y;
    int r = r0 + y;
    int c = c0 + threadIdx.x;
    if (c >= N) return;

    if (r < M) {
        tile[y][threadIdx.x ^ y] = A[r * N + c];
    }

    y += blockDim.y;
    r += blockDim.y;
    if (r < M) {
        tile[y][threadIdx.x ^ y] = A[r * N + c];
    }

    y += blockDim.y;
    r += blockDim.y;
    if (r < M) {
        tile[y][threadIdx.x ^ y] = A[r * N + c];
    }

    y += blockDim.y;
    r += blockDim.y;
    if (r < M) {
        tile[y][threadIdx.x ^ y] = A[r * N + c];
    }

    __syncthreads();  // 同步线程块

    /* -------- 写入阶段 -------- */
    // (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
    // thread y 方向负责：矩阵 B 的行，shared memory 的列
    // thread x 方向负责：矩阵 B 的列，shared memory 的行
    // shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
    r = r0 + threadIdx.x;
    if (r >= M) return;

    y = threadIdx.y;
    c = c0 + y;
    if (c < N) {
        B[c * M + r] = tile[threadIdx.x][threadIdx.x ^ y];
    }

    y += blockDim.y;
    c += blockDim.y;
    if (c < N) {
        B[c * M + r] = tile[threadIdx.x][threadIdx.x ^ y];
    }

    y += blockDim.y;
    c += blockDim.y;
    if (c < N) {
        B[c * M + r] = tile[threadIdx.x][threadIdx.x ^ y];
    }

    y += blockDim.y;
    c += blockDim.y;
    if (c < N) {
        B[c * M + r] = tile[threadIdx.x][threadIdx.x ^ y];
    }
}

// A[r,c] = B[c,r] => B[c*M + r] = A[r*N + c]
void host_transpose(float* A, float* B, const int M, const int N) {
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            B[c * M + r] = A[r * N + c];
        }
    }
}
