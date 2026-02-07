#include <cuda_runtime.h>
#include <iostream>
#include "cmdline.h"
#include "cuda_utils.cuh"
#include "data_utils.h"
#include "elementwise.h"

int main(int argc, char* argv[]) {
    cmdline::parser args;
    args.add<int>("kernel", 'k', "which kernel to use (0: naive, 1: optimized)", false, 0,
                  cmdline::range(0, 3));
    args.add<int>("N", 'n', "number of elements", false, 16 * 1024 * 1024);
    args.add<int>("block", 'b', "block size", false, 256);
    args.add<int>("grid", 'g', "grid size", false, 0);
    args.add<int>("device", 'd', "gpu id", false, 0);
    args.parse_check(argc, argv);

    const int N = args.get<int>("N");
    const int nbytes = N * sizeof(float);
    const int whichKernel = args.get<int>("kernel");
    const int blockSize = args.get<int>("block");
    const int gridSize = args.get<int>("grid") == 0 ? (N + blockSize - 1) / blockSize
                                                    : args.get<int>("grid");
    const int deviceID = args.get<int>("device");

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceID));
    std::cout << "CUDA device [" << deviceID << "]: [" << deviceProp.name << "]"
              << std::endl;

    float* A = nullptr;
    checkCudaErrors(cudaMallocHost((void**)&A, nbytes));
    initialRangeData(A, N, 0.0f, 0.01f);
    float* B = nullptr;
    checkCudaErrors(cudaMallocHost((void**)&B, nbytes));
    initialRangeData(B, N, 0.0f, -0.02f);
    float* C = nullptr;
    checkCudaErrors(cudaMallocHost((void**)&C, nbytes));
    memset(C, 0, nbytes);

    // 数据拷贝: host -> device
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_A, nbytes));
    checkCudaErrors(cudaMalloc((void**)&d_B, nbytes));
    checkCudaErrors(cudaMalloc((void**)&d_C, nbytes));
    checkCudaErrors(cudaMemcpy(d_A, A, nbytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, nbytes, cudaMemcpyHostToDevice));

    // 调用 host
    host_add_fp32(A, B, C, N);

    // 调用 device kernel
    kernel_add_fp32(whichKernel, blockSize, gridSize, d_A, d_B, d_C, N);
    getLastCudaError("kernel add fp32 failed");
    cudaDeviceSynchronize();

    // 检查结果
    float* gpuC = nullptr;
    checkCudaErrors(cudaMallocHost((void**)&gpuC, nbytes));
    cudaMemcpy(gpuC, d_C, nbytes, cudaMemcpyDeviceToHost);

    if (checkResult(C, gpuC, N)) {
        printf("Correct result\n");
    } else {
        printf("Incorrect result\n");
    }

    // 释放资源
    checkCudaErrors(cudaFreeHost(A));
    checkCudaErrors(cudaFreeHost(B));
    checkCudaErrors(cudaFreeHost(C));
    checkCudaErrors(cudaFreeHost(gpuC));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return 0;
}