#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include "cublas_handle.h"
#include "cuda_utils.cuh"

namespace cuda_learning {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static inline cublasHandle_t to_handle(void* p) {
    return reinterpret_cast<cublasHandle_t>(p);
}

static inline void check_cublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error ") + std::to_string(status) +
                                 " @ " + file + ":" + std::to_string(line));
    }
}

#define CUBLAS_CHECK(stmt) check_cublas((stmt), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// CublasHandle implementation
// ---------------------------------------------------------------------------

CublasHandle::CublasHandle() {
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    handle_ = reinterpret_cast<void*>(h);
}

CublasHandle::~CublasHandle() {
    if (handle_) {
        cublasDestroy(to_handle(handle_));
        handle_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// SGEMM — C = alpha * A * B + beta * C  (all row-major)
//
// cuBLAS uses column-major convention internally.  To compute the row-major
// product  C(M×N) = A(M×K) * B(K×N)  we exploit the equivalence:
//
//   row-major C = A * B
//   ≡  col-major C^T = B^T * A^T
//
// In cuBLAS terms:
//   cublasSgemm(handle,
//     CUBLAS_OP_N, CUBLAS_OP_N,   // no-transpose B^T, no-transpose A^T
//     N, M, K,                    // dims of output (N cols, M rows in col-major)
//     &alpha,
//     d_B, N,                     // B^T in col-major is B in row-major, leading dim N
//     d_A, K,                     // A^T in col-major is A in row-major, leading dim K
//     &beta,
//     d_C, N);                    // C^T output, leading dim N
// ---------------------------------------------------------------------------
void CublasHandle::sgemm(const float* d_A, const float* d_B, float* d_C, int M, int K,
                         int N, float alpha, float beta) const {
    CUBLAS_CHECK(cublasSgemm(to_handle(handle_), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, d_B, N, d_A, K, &beta, d_C, N));
}

// ---------------------------------------------------------------------------
// Transpose — d_B = d_A^T  (row-major A: M×N → B: N×M)
//
// Uses cublasSgeam:  B = alpha * op(A) + beta * op(C)
//   with alpha=1, beta=0, op=TRANSPOSE, so  B = A^T.
//
// cuBLAS column-major interpretation:
//   d_A is treated as col-major A_cm (N rows, M cols) with leading dim N.
//   Transposing col-major (N×M) → (M×N) col-major result = row-major B (N×M).
// ---------------------------------------------------------------------------
void CublasHandle::sgeam_transpose(const float* d_A, float* d_B, int M, int N) const {
    const float one = 1.0f;
    const float zero = 0.0f;
    // cublasSgeam: B = alpha * A^T  (col-major: A is N×M → B is M×N)
    CUBLAS_CHECK(cublasSgeam(to_handle(handle_), CUBLAS_OP_T, CUBLAS_OP_N, M,
                             N,             // rows/cols of output in col-major
                             &one, d_A, N,  // A in col-major: N rows, M cols, lda=N
                             &zero, d_B,
                             M,         // dummy C (beta=0), same shape as output, ldc=M
                             d_B, M));  // output B, ldb=M
}

}  // namespace cuda_learning
