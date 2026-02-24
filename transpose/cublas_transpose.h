#pragma once

// ---------------------------------------------------------------------------
// cuBLAS transpose wrapper
//
// Because cublas_v2.h declares CUDA types, this header may be included from
// regular C++ translation units (no device code required here).
// The actual cublasSgeam call lives in cublas_transpose.cu (compiled by nvcc).
// ---------------------------------------------------------------------------

/// RAII owner for a cublasHandle_t.  Create once; call transpose() repeatedly.
/// Creating / destroying a cuBLAS handle is expensive, so callers should keep
/// one instance alive for the duration of the benchmark.
struct CublasHandle {
    CublasHandle();
    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    /// Compute d_B = A^T where A is a row-major M×N float matrix.
    /// d_A is not modified; d_B must hold at least N*M floats.
    void transpose(const float* d_A, float* d_B, int M, int N) const;

   private:
    void* handle_;  // opaque cublasHandle_t; avoids pulling cublas_v2.h into callers
};
