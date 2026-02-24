#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include "cublas_transpose.h"

// ---------------------------------------------------------------------------
// Error checking helpers (local to this TU)
// ---------------------------------------------------------------------------

static void cublas_check(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(status), file,
                     line);
        std::exit(EXIT_FAILURE);
    }
}
#define checkCublasErrors(val) cublas_check((val), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// CublasHandle implementation
// ---------------------------------------------------------------------------

CublasHandle::CublasHandle() {
    cublasHandle_t h{};
    checkCublasErrors(cublasCreate(&h));
    handle_ = static_cast<void*>(h);
}

CublasHandle::~CublasHandle() {
    if (handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(handle_));
        handle_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// transpose
//
// Key insight: a row-major M×N matrix stored in d_A has the same byte layout
// as a column-major N×M matrix.  We INTERPRET d_A as a column-major N×M
// matrix (lda = N rows → leading dim = N).  Applying CUBLAS_OP_T produces a
// column-major M×N output (ldc = M), whose byte layout is row-major N×M —
// exactly the transposed result we want.
//
//   cublasSgeam(handle, OP_T, OP_N,   M,   N,
//               alpha,  d_A,  lda=N,
//               beta,   null, M,
//               d_B,    ldc=M)
// ---------------------------------------------------------------------------

void CublasHandle::transpose(const float* d_A, float* d_B, int M, int N) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    checkCublasErrors(cublasSgeam(static_cast<cublasHandle_t>(handle_), CUBLAS_OP_T,
                                  CUBLAS_OP_N, M, N, &alpha, d_A,
                                  N,                  // A_cuBLAS: N×M col-major (lda=N)
                                  &beta, nullptr, M,  // B unused (beta=0)
                                  d_B, M));  // C: M×N col-major (ldc=M) = N×M row-major
}
