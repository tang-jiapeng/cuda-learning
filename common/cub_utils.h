#pragma once

#include <cstddef>

// ---------------------------------------------------------------------------
// cub_utils.h — Thin CUB wrappers shared across all operators
//
// Splitting the two-step CUB device-reduce pattern into helpers avoids
// exposing the CUB headers to every caller.
//
// Usage:
//   std::size_t scratch = cub_reduce_scratch_bytes(N);
//   // allocate d_temp of `scratch` bytes on device
//   cub_reduce_sum(d_temp, scratch, d_input, d_out, N);
// ---------------------------------------------------------------------------

namespace cuda_learning {

/// Returns the number of bytes of temporary device storage required for
/// a CUB DeviceReduce::Sum over N float elements.
std::size_t cub_reduce_scratch_bytes(int N);

/// Runs CUB DeviceReduce::Sum.
///   d_temp      — pre-allocated temp storage (size >= cub_reduce_scratch_bytes(N))
///   temp_bytes  — size of d_temp in bytes
///   d_input     — device array of N floats
///   d_out       — device scalar result (single float)
///   N           — number of elements
void cub_reduce_sum(void* d_temp, std::size_t temp_bytes, const float* d_input,
                    float* d_out, int N);

}  // namespace cuda_learning
