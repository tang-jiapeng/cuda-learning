#pragma once

#include <functional>
#include <string>

std::string get_kernel_name(int kernel_num);

void host_add_fp32(float* A, float* B, float* C, int N);

void launch_elementwise_add_kernel(int whichKernel, int blockSize, int gridSize, 
                   float* d_A, float* d_B, float* d_C, int N);