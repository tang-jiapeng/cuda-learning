#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "args.h"
#include "benchmark.h"
#include "cub_reduce.h"
#include "cuda_utils.cuh"
#include "data_utils.h"
#include "memory.h"
#include "reduce.h"

using namespace cuda_learning;

// ---------------------------------------------------------------------------
// Tolerance check for floating-point sums (allow 0.1% relative + small abs)
// ---------------------------------------------------------------------------
static bool close_enough(float a, float b) {
    return std::abs(a - b) <= std::abs(b) * 1e-3f + 1e-2f;
}

int main(int argc, char* argv[]) {
    // ---- Argument Parsing ----
    ArgParser args;
    args.add("kernel", "-1", "kernel index (-1: run all, 0-9: specific kernel)")
        .add("N", "16777216", "number of elements (default: 16 M)")
        .add("seed", "42", "random seed for data generation")
        .add("device", "0", "CUDA device id");

    if (!args.parse(argc, argv)) return 0;

    const int user_kernel = args.get<int>("kernel");
    const int N = args.get<int>("N");
    const unsigned seed = args.get<unsigned>("seed");
    const int device_id = args.get<int>("device");

    // ---- Device Setup ----
    checkCudaErrors(cudaSetDevice(device_id));
    print_device_info(device_id);
    std::cout << "N=" << N << "  seed=" << seed << "\n\n";

    // ---- Host Buffers ----
    HostBuffer<float> h_A(N);
    fill_normal(h_A.get(), static_cast<std::size_t>(N), 0.0f, 1.0f, seed);

    const float cpu_sum = host_reduce_sum(h_A.get(), N);
    std::printf("CPU sum (pairwise) = %.6f\n\n", cpu_sum);

    // ---- Device Buffers ----
    // d_pristine: never modified — source for reset between iterations
    // d_work:     scratch buffer that kernels reduce in-place
    DeviceBuffer<float> d_pristine(N), d_work(N);
    d_pristine.upload(h_A.get());

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);

    // ---- Peak bandwidth ----
    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
    const double peak_bw =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    // ---- Verify Lambda ----
    // result is in d_work[0] after all reduction stages complete
    auto verify = [&]() -> bool {
        float gpu_sum = 0.0f;
        checkCudaErrors(
            cudaMemcpy(&gpu_sum, d_work.get(), sizeof(float), cudaMemcpyDeviceToHost));
        const bool ok = close_enough(gpu_sum, cpu_sum);
        if (!ok)
            std::printf("  [MISMATCH] cpu=%.6f  gpu=%.6f  diff=%.6f\n", cpu_sum, gpu_sum,
                        std::abs(gpu_sum - cpu_sum));
        return ok;
    };

    // ---- Reset Lambda ----
    // Restore d_work from d_pristine before each timed iteration so that
    // in-place kernels always see the original data.
    auto reset = [&] {
        checkCudaErrors(
            cudaMemcpy(d_work.get(), d_pristine.get(), bytes, cudaMemcpyDeviceToDevice));
    };

    // ---- Run Custom Kernels ----
    constexpr int n_kernels = 10;
    std::vector<BenchmarkResult> results;

    BenchmarkConfig cfg;
    cfg.reset_fn = reset;

    auto run_one = [&](int k) {
        results.push_back(run_benchmark(
            get_kernel_name(k), [&] { launch_reduce_sum_kernel(k, d_work.get(), N); },
            verify, bytes, cfg));
    };

    if (user_kernel == -1) {
        for (int k = 0; k < n_kernels; ++k) run_one(k);
    } else {
        run_one(user_kernel);
    }

    // ---- CUB DeviceReduce::Sum Baseline ----
    if (user_kernel == -1) {
        DeviceBuffer<float> d_out(1);
        const std::size_t temp_bytes = cub_reduce_scratch_bytes(N);
        void* d_temp = nullptr;
        checkCudaErrors(cudaMalloc(&d_temp, temp_bytes));

        auto cub_verify = [&]() -> bool {
            float gpu_sum = 0.0f;
            checkCudaErrors(
                cudaMemcpy(&gpu_sum, d_out.get(), sizeof(float), cudaMemcpyDeviceToHost));
            const bool ok = close_enough(gpu_sum, cpu_sum);
            if (!ok)
                std::printf("  [MISMATCH] cpu=%.6f  gpu=%.6f  diff=%.6f\n", cpu_sum,
                            gpu_sum, std::abs(gpu_sum - cpu_sum));
            return ok;
        };

        // CUB reads d_pristine (read-only), no reset needed
        results.push_back(run_benchmark(
            "cub::DeviceReduce::Sum",
            [&] { cub_reduce_sum(d_temp, temp_bytes, d_pristine.get(), d_out.get(), N); },
            cub_verify, bytes));

        checkCudaErrors(cudaFree(d_temp));
    }

    print_results(results, peak_bw);
    return 0;
}
