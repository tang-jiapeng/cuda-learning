#pragma once

#include <string>

std::string get_kernel_name(int kernel_num);

void host_gemm(float* A, float* B, float* out, const int M, const int K, const int N);

void lauch_gemm_kernel(int whichKernel, float* A, float* B, float* out, const int M,
                       const int K, const int N);