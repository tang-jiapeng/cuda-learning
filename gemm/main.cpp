#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "args.h"
#include "benchmark.h"
#include "cuda_utils.cuh"
#include "data_utils.h"
#include "gemm.h"
#include "memory.h"

using namespace cuda_learning;

int main(int argc, char* argv[]) {
    // ---- Argument Parsing ----
    ArgParser args;
    args.add("kernel", "-1", "kernel index (-1: run all, 0-2: specific kernel)")
        .add("M", "2048", "matrix A rows (default: 2048)")
        .add("K", "2048", "matrix A cols / matrix B rows (default: 2048)")
        .add("N", "2048", "matrix B cols (default: 2048)")
        .add("device", "0", "CUDA device id");

    if (!args.parse(argc, argv)) return 0;

    const int user_kernel = args.get<int>("kernel");
    const int M = args.get<int>("M");
    const int K = args.get<int>("K");
    const int N = args.get<int>("N");
    const int device_id = args.get<int>("device");

    // ---- Device Setup ----
    checkCudaErrors(cudaSetDevice(device_id));
    print_device_info(device_id);
    std::cout << "M=" << M << "  K=" << K << "  N=" << N << "\n\n";

    // ---- Host Buffers ----
    const std::size_t sizeA = static_cast<std::size_t>(M) * K;
    const std::size_t sizeB = static_cast<std::size_t>(K) * N;
    const std::size_t sizeOut = static_cast<std::size_t>(M) * N;

    HostBuffer<float> h_A(sizeA), h_B(sizeB), h_ref(sizeOut);
    fill_uniform(h_A.get(), sizeA, -1.0f, 1.0f);
    fill_uniform(h_B.get(), sizeB, -1.0f, 1.0f);
    host_gemm(h_A.get(), h_B.get(), h_ref.get(), M, K, N);

    // ---- Device Buffers ----
    DeviceBuffer<float> d_A(sizeA), d_B(sizeB), d_out(sizeOut);
    d_A.upload(h_A.get());
    d_B.upload(h_B.get());
    d_out.zero();

    // ---- Verify Lambda ----
    HostBuffer<float> h_out(sizeOut);
    auto verify = [&]() -> bool {
        d_out.download(h_out.get());
        return check_result(h_ref.get(), h_out.get(), sizeOut);
    };

    // GEMM reads A and B once, writes out once
    const std::size_t bytes = 2ULL * (sizeA + sizeB + sizeOut) * sizeof(float);

    // ---- Peak bandwidth ----
    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
    const double peak_bw =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    // ---- Run Custom Kernels ----
    constexpr int n_kernels = 3;
    std::vector<BenchmarkResult> results;

    auto run_one = [&](int k) {
        checkCudaErrors(cudaMemset(d_out.get(), 0, d_out.bytes()));
        results.push_back(run_benchmark(
            get_kernel_name(k),
            [&] { lauch_gemm_kernel(k, d_A.get(), d_B.get(), d_out.get(), M, K, N); },
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