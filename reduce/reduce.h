#pragma once

#include <string>

std::string get_kernel_name(int kernel_num);

float host_reduce_sum(float* input, const int N);

void launch_reduce_sum_kernel(int whichKernel, float* input, const int N);
