#include <cub/cub.cuh>
#include "cub_utils.h"
#include "cuda_utils.cuh"

namespace cuda_learning {

std::size_t cub_reduce_scratch_bytes(int N) {
    std::size_t bytes = 0;
    checkCudaErrors(cub::DeviceReduce::Sum(nullptr, bytes, (const float*)nullptr,
                                           (float*)nullptr, N));
    return bytes;
}

void cub_reduce_sum(void* d_temp, std::size_t temp_bytes, const float* d_input,
                    float* d_out, int N) {
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_input, d_out, N));
}

}  // namespace cuda_learning
