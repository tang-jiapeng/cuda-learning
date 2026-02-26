#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "args.h"
#include "benchmark.h"
#include "cublas_handle.h"
#include "cuda_utils.cuh"
#include "data_utils.h"
#include "gemm.h"
#include "memory.h"

using namespace cuda_learning;

int main(int argc, char* argv[]) {
    // ---- Argument Parsing ----
    ArgParser args;
    args.add("kernel", "-1", "kernel index (-1: run all, 0-N: specific kernel)")
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

    HostBuffer<float> h_A(sizeA), h_B(sizeB);
    fill_uniform(h_A.get(), sizeA, -1.0f, 1.0f);
    fill_uniform(h_B.get(), sizeB, -1.0f, 1.0f);

    // ---- Device Buffers ----
    DeviceBuffer<float> d_A(sizeA), d_B(sizeB), d_out(sizeOut), d_ref(sizeOut);
    d_A.upload(h_A.get());
    d_B.upload(h_B.get());

    // ---- GPU Reference via cuBLAS (avoids slow CPU triple-loop) ----
    CublasHandle cublas;
    cublas.sgemm(d_A.get(), d_B.get(), d_ref.get(), M, K, N);
    checkCudaErrors(cudaDeviceSynchronize());

    HostBuffer<float> h_ref(sizeOut);
    d_ref.download(h_ref.get());

    // ---- Verify Lambda ----
    // atol scales with K: fp32 sum of K terms accumulates O(K*eps) rounding
    // error (~1.2e-7 per op). Use K*1e-6 as a conservative safe threshold.
    const float gemm_atol = static_cast<float>(K) * 1e-6f;
    HostBuffer<float> h_out(sizeOut);
    auto verify = [&]() -> bool {
        d_out.download(h_out.get());
        return check_result(h_ref.get(), h_out.get(), sizeOut, gemm_atol);
    };

    // GEMM reads A (M*K) and B (K*N) once, writes C (M*N) once
    const std::size_t bytes = (sizeA + sizeB + sizeOut) * sizeof(float);

    // ---- Device Properties ----
    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));

    // Peak FP32 TFLOPS: clockRate(kHz) * SM_count * FP32_cores_per_SM * 2 (FMA) / 1e9
    // FP32 cores/SM by compute capability major.minor:
    //   8.0 (A100): 64    8.6 (RTX 30xx): 128    9.0 (RTX 40xx): 128
    //   7.x (Turing/Volta): 64
    auto fp32_per_sm = [](int major, int minor) -> int {
        if (major == 9) return 128;
        if (major == 8) return (minor == 0) ? 64 : 128;  // A100 vs RTX 30xx
        if (major == 7) return 64;
        return 64;  // conservative fallback
    };
    const int cores_per_sm = fp32_per_sm(prop.major, prop.minor);
    const double peak_tflops = static_cast<double>(prop.clockRate) * 1e3  // Hz
                               * prop.multiProcessorCount * cores_per_sm *
                               2  // FMA = 2 FLOPs
                               / 1e12;

    // ---- Run Custom Kernels ----
    constexpr int n_kernels = 2;
    std::vector<BenchmarkResult> results;

    auto run_one = [&](int k) {
        d_out.zero();
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

    // ---- cuBLAS Baseline ----
    {
        d_ref.zero();
        auto cublas_bench = run_benchmark(
            "cuBLAS sgemm",
            [&] { cublas.sgemm(d_A.get(), d_B.get(), d_ref.get(), M, K, N); },
            []() -> bool { return true; },  // cuBLAS is the reference — always pass
            bytes);
        results.push_back(cublas_bench);
    }

    // GEMM is compute-bound: use cuBLAS as 100% baseline for relative efficiency
    print_results(results, 0.0, "cuBLAS sgemm");

    // ---- GEMM FLOPS Table ----
    // FLOPs per call = 2 * M * K * N  (one multiply-add per element per k-step)
    const double flops_per_call = 2.0 * M * K * N;

    constexpr int kW0 = 45, kW1 = 10, kW2 = 14, kW3 = 12;
    const std::string sep(kW0 + kW1 + kW2 + kW3 + 3, '-');
    std::cout << "\n" << sep << "\n";
    std::cout << std::left << std::setw(kW0) << "Kernel" << std::right << std::setw(kW1)
              << "TFLOPS" << std::setw(kW2) << "Time (ms)" << std::setw(kW3) << "Compute%"
              << "\n";
    std::cout << sep << "\n";
    for (const auto& r : results) {
        double tflops = flops_per_call / (r.avg_ms / 1000.0) / 1e12;
        double compute_pct = tflops / peak_tflops * 100.0;
        std::cout << std::left << std::setw(kW0) << r.name << std::right << std::fixed
                  << std::setprecision(2) << std::setw(kW1) << tflops << std::setw(kW2)
                  << r.avg_ms << std::setw(kW3) << compute_pct << "\n";
    }
    std::cout << sep << "\n";
    std::cout << "Peak FP32: " << std::fixed << std::setprecision(2) << peak_tflops
              << " TFLOPS\n\n";

    return 0;
}