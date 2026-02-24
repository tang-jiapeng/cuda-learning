#pragma once
#include <cstddef>

// ---------------------------------------------------------------------------
// CUB DeviceReduce::Sum wrappers
//
// Because CUB headers contain device code, they must be compiled by nvcc.
// These wrapper functions expose a plain C++ interface that can be called
// from regular host (.cpp) translation units.
// ---------------------------------------------------------------------------

/// Returns the number of bytes of temporary device storage needed.
std::size_t cub_reduce_scratch_bytes(int N);

/// Run cub::DeviceReduce::Sum (asynchronous, on the default CUDA stream).
///   d_temp      — pre-allocated device scratch (size >= cub_reduce_scratch_bytes(N))
///   temp_bytes  — size of d_temp in bytes
///   d_input     — input device array  (not modified)
///   d_out       — single-element device output
void cub_reduce_sum(void* d_temp, std::size_t temp_bytes, const float* d_input,
                    float* d_out, int N);
