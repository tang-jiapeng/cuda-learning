#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include "args.h"
#include "benchmark.h"
#include "cuda_utils.cuh"
#include "data_utils.h"
#include "elementwise.h"
#include "memory.h"


using namespace cuda_learning;

int main(int argc, char* argv[]) {
    // ---- Argument Parsing ----
    ArgParser args;
    args.add("kernel", "-1", "kernel index (-1: run all, 0-3: specific kernel)")
        .add("N", "16777216", "number of elements (default: 16 M)")
        .add("block", "256", "thread block size")
        .add("grid", "0", "grid size (0 = auto-computed)")
        .add("device", "0", "CUDA device id");

    if (!args.parse(argc, argv)) return 0;

    const int user_kernel = args.get<int>("kernel");
    const int N = args.get<int>("N");
    const int block_size = args.get<int>("block");
    int grid_size = args.get<int>("grid");
    const int device_id = args.get<int>("device");

    // ---- Device Setup ----
    checkCudaErrors(cudaSetDevice(device_id));
    print_device_info(device_id);

    if (grid_size == 0) grid_size = (N + block_size - 1) / block_size;
    std::cout << "N=" << N << "  block=" << block_size << "  grid=" << grid_size
              << "\n\n";

    // ---- Pinned Host Buffers ----
    HostBuffer<float> h_A(N), h_B(N), h_ref(N);
    fill_range(h_A.get(), static_cast<std::size_t>(N), 0.0f, 0.01f);
    fill_range(h_B.get(), static_cast<std::size_t>(N), 0.0f, -0.02f);

    // CPU reference result
    host_add_fp32(h_A.get(), h_B.get(), h_ref.get(), N);

    // ---- Device Buffers ----
    DeviceBuffer<float> d_A(N), d_B(N), d_C(N);
    d_A.upload(h_A.get());
    d_B.upload(h_B.get());

    // ---- Shared verify lambda (downloads d_C, checks against h_ref) ----
    HostBuffer<float> h_out(N);
    auto verify = [&]() -> bool {
        d_C.download(h_out.get());
        return check_result(h_ref.get(), h_out.get(), static_cast<std::size_t>(N));
    };

    // bytes read + written by a single kernel launch (2 reads + 1 write)
    const std::size_t bytes = 3ULL * static_cast<std::size_t>(N) * sizeof(float);

    // ---- Peak bandwidth for roofline column ----
    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
    const double peak_bw =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    // ---- Run Benchmarks ----
    constexpr int n_kernels = 4;
    std::vector<BenchmarkResult> results;

    auto run_one = [&](int k) {
        results.push_back(run_benchmark(
            get_kernel_name(k),
            [&] {
                launch_elementwise_add_kernel(k, block_size, grid_size, d_A.get(),
                                              d_B.get(), d_C.get(), N);
            },
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