#pragma once

#include <functional>
#include <string>

std::string get_kernel_name(int kernel_num);

void host_transpose(float* A, float* B,const int M, const int N);

void launch_transpose_kernel(int whichKernel, float* A, float* B, const int M,
                             const int N, int blockDimX);