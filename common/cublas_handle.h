#pragma once

// ---------------------------------------------------------------------------
// cublas_handle.h — General-purpose cuBLAS RAII wrapper (common infrastructure)
//
// This header may be included from plain C++ (.cpp) translation units.
// All device-code dependencies live in cublas_handle.cu (compiled by nvcc).
//
// Supported operations (all row-major float):
//   sgemm            — C = alpha * A * B + beta * C   (GEMM)
//   sgeam_transpose  — B = A^T                        (in-place transpose)
// ---------------------------------------------------------------------------

namespace cuda_learning {

/// RAII owner for a cublasHandle_t.
/// Handle creation/destruction is expensive — create once per program and reuse.
class CublasHandle {
   public:
    CublasHandle();
    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    // ---------------------------------------------------------------------------
    // SGEMM: C = alpha * A * B + beta * C
    //   A  — row-major M × K
    //   B  — row-major K × N
    //   C  — row-major M × N (output)
    // ---------------------------------------------------------------------------
    void sgemm(const float* d_A, const float* d_B, float* d_C, int M, int K, int N,
               float alpha = 1.0f, float beta = 0.0f) const;

    // ---------------------------------------------------------------------------
    // Transpose: d_B = d_A^T
    //   d_A — row-major M × N
    //   d_B — row-major N × M  (output, must hold at least N*M floats)
    // ---------------------------------------------------------------------------
    void sgeam_transpose(const float* d_A, float* d_B, int M, int N) const;

   private:
    void* handle_;  // opaque cublasHandle_t; avoids pulling cublas_v2.h into callers
};

}  // namespace cuda_learning
