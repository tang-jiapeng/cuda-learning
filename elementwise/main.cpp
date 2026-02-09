#include <cuda_runtime.h>
#include <iostream>
#include "cmdline.h"
#include "cuda_utils.cuh"
#include "data_utils.h"
#include "elementwise.h"

// --- 辅助函数：分配和初始化内存 ---
void init_host_data(float** ptr, int N, float start, float step) {
    checkCudaErrors(cudaMallocHost((void**)ptr, N * sizeof(float)));
    initialRangeData(*ptr, N, start, step);
}

// --- 辅助函数：运行单个 Benchmark ---
void run_benchmark(int kernel_idx, int N, int blockSize, int gridSize, float* h_A,
                   float* h_B, float* h_ref,            // Host data
                   float* d_A, float* d_B, float* d_C)  // Device data
{
    std::string name = get_kernel_name(kernel_idx);
    printf("----------------------------------------------------------------\n");
    printf("Running Kernel [%d]: %s\n", kernel_idx, name.c_str());

    // 预热 (Warm-up)
    // 不计入时间，跑 10 次
    for (int i = 0; i < 10; ++i) {
        launch_elementwise_add_kernel(kernel_idx, blockSize, gridSize, d_A, d_B, d_C, N);
    }
    checkCudaErrors(cudaDeviceSynchronize());  // 确保预热完成

    // 准备计时器
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // 正式测试 (Repeated Run)
    const int n_repeats = 100;  // 循环运行 100 次取平均

    checkCudaErrors(cudaEventRecord(start));

    for (int i = 0; i < n_repeats; ++i) {
        launch_elementwise_add_kernel(kernel_idx, blockSize, gridSize, d_A, d_B, d_C, N);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    getLastCudaError("Kernel execution failed");

    // 计算耗时 (总时间 / 次数)
    float total_milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&total_milliseconds, start, stop));
    float avg_milliseconds = total_milliseconds / n_repeats;

    // 校验结果 (只需要校验最后一次运行的结果)
    float* h_C = nullptr;
    checkCudaErrors(cudaMallocHost((void**)&h_C, N * sizeof(float)));
    checkCudaErrors(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool is_correct = checkResult(h_ref, h_C, N);

    // 打印结果
    double bandwidth = (3.0 * N * sizeof(float)) / (avg_milliseconds / 1000.0) / 1e9;

    printf("Status: %s\n", is_correct ? "PASS" : "FAIL");
    printf("Time  : %.4f ms (Average of %d runs)\n", avg_milliseconds, n_repeats);
    printf("BW    : %.2f GB/s\n", bandwidth);

    // 清理
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(h_C));
}

int main(int argc, char* argv[]) {
    cmdline::parser args;
    args.add<int>("kernel", 'k', "kernel type (-1: run all, 0-3: specific)", false, -1,
                  cmdline::range(-1, 3));
    args.add<int>("N", 'n', "number of elements", false, 16 * 1024 * 1024);
    args.add<int>("block", 'b', "block size", false, 256);
    args.add<int>("grid", 'g', "grid size (0 for auto)", false, 0);
    args.add<int>("device", 'd', "gpu id", false, 0);
    args.parse_check(argc, argv);

    const int N = args.get<int>("N");
    const int user_kernel = args.get<int>("kernel");
    const int blockSize = args.get<int>("block");
    const int gridSize = args.get<int>("grid");
    const int deviceID = args.get<int>("device");
    const size_t nbytes = N * sizeof(float);

    // --- 设置设备 ---
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceID));
    checkCudaErrors(cudaSetDevice(deviceID));
    std::cout << "Device: " << deviceProp.name << ", N: " << N << " elements"
              << std::endl;

    // --- 准备 Host 数据 ---
    float *h_A = nullptr, *h_B = nullptr, *h_ref = nullptr;
    init_host_data(&h_A, N, 0.0f, 0.01f);
    init_host_data(&h_B, N, 0.0f, -0.02f);

    // 计算 CPU 参考结果 (Reference)
    checkCudaErrors(cudaMallocHost((void**)&h_ref, nbytes));
    host_add_fp32(h_A, h_B, h_ref, N);

    // --- 准备 Device 数据 ---
    // 只需要分配一次，拷贝一次输入数据，所有 Benchmark 共用
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_A, nbytes));
    checkCudaErrors(cudaMalloc((void**)&d_B, nbytes));
    checkCudaErrors(cudaMalloc((void**)&d_C, nbytes));

    checkCudaErrors(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice));

    // --- 运行 Benchmark ---
    if (user_kernel == -1) {
        // 比较模式：循环运行所有实现
        std::cout << "\n>>> Starting Comparison Benchmark <<<\n";
        for (int k = 0; k <= 3; ++k) {
            run_benchmark(k, N, blockSize, gridSize, h_A, h_B, h_ref, d_A, d_B, d_C);
        }
    } else {
        // 单次运行模式
        run_benchmark(user_kernel, N, blockSize, gridSize, h_A, h_B, h_ref, d_A, d_B,
                      d_C);
    }

    // --- 释放资源 ---
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_ref));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return 0;
}