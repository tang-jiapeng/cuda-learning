#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "args.h"
#include "benchmark.h"
#include "cuda_utils.cuh"
#include "data_utils.h"
#include "memory.h"
#include "transpose.h"

using namespace cuda_learning;

int main(int argc, char* argv[]) {
    // ---- Argument Parsing ----
    ArgParser args;
    args.add("kernel", "-1", "kernel index (-1: run all, 0-7: specific kernel)")
        .add("M", "4096", "matrix rows (default: 4096)")
        .add("N", "4096", "matrix cols (default: 4096)")
        .add("block", "16", "thread block dim for naive kernels (block x block)")
        .add("device", "0", "CUDA device id");

    if (!args.parse(argc, argv)) return 0;

    const int user_kernel = args.get<int>("kernel");
    const int M = args.get<int>("M");
    const int N = args.get<int>("N");
    const int block_dim = args.get<int>("block");
    const int device_id = args.get<int>("device");

    // ---- Device Setup ----
    checkCudaErrors(cudaSetDevice(device_id));
    print_device_info(device_id);
    std::cout << "Matrix: " << M << " x " << N << "  naive-block=" << block_dim << "\n\n";

    // ---- Host Buffers ----
    // A: M x N,  B (result): N x M
    const std::size_t sizeA = static_cast<std::size_t>(M) * N;
    const std::size_t sizeB = static_cast<std::size_t>(N) * M;  // same, just clearer

    HostBuffer<float> h_A(sizeA), h_ref(sizeB);
    fill_uniform(h_A.get(), sizeA, -1.0f, 1.0f);
    host_transpose(h_A.get(), h_ref.get(), M, N);

    // ---- Device Buffers ----
    DeviceBuffer<float> d_A(sizeA), d_B(sizeB);
    d_A.upload(h_A.get());

    // ---- Verify Lambda ----
    HostBuffer<float> h_out(sizeB);
    auto verify = [&]() -> bool {
        d_B.download(h_out.get());
        return check_result(h_ref.get(), h_out.get(), sizeB);
    };

    // Transpose reads A once and writes B once
    const std::size_t bytes = 2ULL * sizeA * sizeof(float);

    // ---- Peak bandwidth ----
    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
    const double peak_bw =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    // ---- Run Benchmarks ----
    constexpr int n_kernels = 8;
    std::vector<BenchmarkResult> results;

    auto run_one = [&](int k) {
        // Reset output buffer before each kernel so verify is independent
        checkCudaErrors(cudaMemset(d_B.get(), 0, sizeB * sizeof(float)));
        results.push_back(run_benchmark(
            get_kernel_name(k),
            [&] { launch_transpose_kernel(k, d_A.get(), d_B.get(), M, N, block_dim); },
            verify, bytes));
    };

    if (user_kernel == -1) {
        for (int k = 0; k < n_kernels; ++k) run_one(k);
    } else {
        run_one(user_kernel);
    }

    print_results(results, peak_bw);
    return 0;
}
