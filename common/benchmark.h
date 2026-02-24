#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include "cuda_utils.cuh"

// All utilities live in the cuda_learning namespace to avoid polluting global scope.
namespace cuda_learning {

// ---------------------------------------------------------------------------
// Configuration & Result Types
// ---------------------------------------------------------------------------

struct BenchmarkConfig {
    int warmup_iters{10};  /// iterations used for warm-up (not counted in time)
    int bench_iters{100};  /// iterations used for timing
    /// Optional: called BEFORE each timed iteration (not included in timing).
    /// Useful when the kernel mutates its input (e.g. in-place reductions) and
    /// the scratch buffer must be restored between iterations.
    std::function<void()> reset_fn;
};

struct BenchmarkResult {
    std::string name;
    bool passed{false};
    float avg_ms{0.0f};
    double bandwidth_GBs{0.0};
};

// ---------------------------------------------------------------------------
// run_benchmark
//
// Template parameters:
//   KernelFn  — callable ()  -> void   (captures all kernel arguments via closure)
//   VerifyFn  — callable ()  -> bool   (returns true when output is correct)
//
// Parameters:
//   name             — display name shown in the results table
//   kernel_fn        — lambda / functor that launches the kernel
//   verify_fn        — lambda / functor that validates the last kernel output
//   bytes_accessed   — total bytes read + written by a single kernel call
//                      (used to compute memory bandwidth)
//   cfg              — optional tuning knobs (warmup / bench iterations)
// ---------------------------------------------------------------------------

template <typename KernelFn, typename VerifyFn>
BenchmarkResult run_benchmark(std::string_view name, KernelFn&& kernel_fn,
                              VerifyFn&& verify_fn, std::size_t bytes_accessed,
                              const BenchmarkConfig& cfg = {}) {
    BenchmarkResult result;
    result.name = std::string(name);

    // ---- Warm-up ----
    // If a reset is required (e.g. in-place kernels), apply it before each call
    // so each warm-up iteration sees a fresh input.
    for (int i = 0; i < cfg.warmup_iters; ++i) {
        if (cfg.reset_fn) cfg.reset_fn();
        kernel_fn();
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // ---- Correctness check (one clean run after warm-up) ----
    if (cfg.reset_fn) cfg.reset_fn();
    kernel_fn();
    checkCudaErrors(cudaDeviceSynchronize());
    result.passed = verify_fn();

    // ---- Timing ----
    cudaEvent_t ev_start{}, ev_stop{};
    checkCudaErrors(cudaEventCreate(&ev_start));
    checkCudaErrors(cudaEventCreate(&ev_stop));

    float total_ms = 0.0f;

    if (cfg.reset_fn) {
        // Per-iteration reset: time each call individually and accumulate.
        for (int i = 0; i < cfg.bench_iters; ++i) {
            cfg.reset_fn();
            checkCudaErrors(cudaEventRecord(ev_start));
            kernel_fn();
            checkCudaErrors(cudaEventRecord(ev_stop));
            checkCudaErrors(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            checkCudaErrors(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            total_ms += ms;
        }
    } else {
        // No reset needed: single contiguous timing region.
        checkCudaErrors(cudaEventRecord(ev_start));
        for (int i = 0; i < cfg.bench_iters; ++i) kernel_fn();
        checkCudaErrors(cudaEventRecord(ev_stop));
        checkCudaErrors(cudaEventSynchronize(ev_stop));
        checkCudaErrors(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    }

    getLastCudaError("Kernel execution failed");

    result.avg_ms = total_ms / static_cast<float>(cfg.bench_iters);
    result.bandwidth_GBs =
        static_cast<double>(bytes_accessed) / (result.avg_ms / 1000.0) / 1e9;

    checkCudaErrors(cudaEventDestroy(ev_start));
    checkCudaErrors(cudaEventDestroy(ev_stop));
    return result;
}

// ---------------------------------------------------------------------------
// print_results — Pretty-print a comparison table
// ---------------------------------------------------------------------------

inline void print_results(const std::vector<BenchmarkResult>& results,
                          double peak_bandwidth_GBs = 0.0) {
    constexpr int kColName = 45;
    constexpr int kColStat = 8;
    constexpr int kColTime = 14;
    constexpr int kColBW = 14;
    constexpr int kColEff = 10;

    const bool show_eff = peak_bandwidth_GBs > 0.0;
    const int total_w =
        kColName + kColStat + kColTime + kColBW + (show_eff ? kColEff : 0) + 4;
    const std::string sep(total_w, '-');

    std::cout << "\n" << sep << "\n";
    std::cout << std::left << std::setw(kColName) << "Kernel" << std::right
              << std::setw(kColStat) << "Status" << std::setw(kColTime) << "Time (ms)"
              << std::setw(kColBW) << "BW (GB/s)";
    if (show_eff) std::cout << std::setw(kColEff) << "Roofline%";
    std::cout << "\n" << sep << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(kColName) << r.name << std::right
                  << std::setw(kColStat) << (r.passed ? "PASS" : "FAIL")
                  << std::setw(kColTime) << std::fixed << std::setprecision(4) << r.avg_ms
                  << std::setw(kColBW) << std::fixed << std::setprecision(2)
                  << r.bandwidth_GBs;
        if (show_eff) {
            const double eff = r.bandwidth_GBs / peak_bandwidth_GBs * 100.0;
            std::cout << std::setw(kColEff) << std::fixed << std::setprecision(1) << eff;
        }
        std::cout << "\n";
    }
    std::cout << sep << "\n";
}

}  // namespace cuda_learning
