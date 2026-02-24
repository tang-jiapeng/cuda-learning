#include <cub/cub.cuh>
#include "cub_reduce.h"
#include "cuda_utils.cuh"

std::size_t cub_reduce_scratch_bytes(int N) {
    void* d_temp = nullptr;
    std::size_t temp_bytes = 0;
    // Passing nullptr for all pointers and the actual N queries the scratch size.
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp, temp_bytes,
                                           static_cast<float*>(nullptr),
                                           static_cast<float*>(nullptr), N));
    return temp_bytes;
}

void cub_reduce_sum(void* d_temp, std::size_t temp_bytes, const float* d_input,
                    float* d_out, int N) {
    // CUB expects a non-const pointer; the data is not modified.
    checkCudaErrors(cub::DeviceReduce::Sum(d_temp, temp_bytes,
                                           const_cast<float*>(d_input), d_out, N));
}
