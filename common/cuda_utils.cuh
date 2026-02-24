#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>


// ---------------------------------------------------------------------------
// Error Checking Macros
// ---------------------------------------------------------------------------

template <typename T>
inline void cuda_check(T result, const char* func, const char* file, int line) {
    if (result) {
        std::fprintf(stderr, "CUDA error at %s:%d  code=%d (%s)  \"%s\"\n", file, line,
                     static_cast<int>(result), cudaGetErrorName(result), func);
        std::exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) cuda_check((val), #val, __FILE__, __LINE__)

inline void cuda_check_last(const char* msg, const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s(%d): %s  (%d) %s.\n", file, line, msg,
                     static_cast<int>(err), cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define getLastCudaError(msg) cuda_check_last(msg, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// CudaTimer — RAII wrapper around cudaEvent timing
// ---------------------------------------------------------------------------

class CudaTimer {
   public:
    CudaTimer() {
        checkCudaErrors(cudaEventCreate(&start_));
        checkCudaErrors(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Non-copyable, movable
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;
    CudaTimer(CudaTimer&&) = default;
    CudaTimer& operator=(CudaTimer&&) = default;

    void start() {
        checkCudaErrors(cudaEventRecord(start_));
    }

    /// Returns elapsed milliseconds since the last start() call.
    float stop() {
        checkCudaErrors(cudaEventRecord(stop_));
        checkCudaErrors(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

   private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// ---------------------------------------------------------------------------
// Device Info
// ---------------------------------------------------------------------------

/// Print key properties of a CUDA device, including peak memory bandwidth.
inline void print_device_info(int device_id = 0) {
    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));

    const double peak_bw_GBs =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    std::printf("==================================================================\n");
    std::printf("Device [%d]: %s\n", device_id, prop.name);
    std::printf("  Compute Capability  : %d.%d\n", prop.major, prop.minor);
    std::printf("  Total Global Memory : %.2f GB\n",
                static_cast<double>(prop.totalGlobalMem) / 1e9);
    std::printf("  SM Count            : %d\n", prop.multiProcessorCount);
    std::printf("  Max Threads / Block : %d\n", prop.maxThreadsPerBlock);
    std::printf("  Warp Size           : %d\n", prop.warpSize);
    std::printf("  Memory Clock Rate   : %.3f GHz\n", prop.memoryClockRate / 1e6);
    std::printf("  Memory Bus Width    : %d bit\n", prop.memoryBusWidth);
    std::printf("  Peak Mem Bandwidth  : %.2f GB/s\n", peak_bw_GBs);
    std::printf("==================================================================\n");
}