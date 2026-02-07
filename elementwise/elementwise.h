#pragma once

void host_add_fp32(const float* A, const float* B, float* C, int N);

void kernel_add_fp32(int whichKernel, int blockSize, int gridSize, float* d_A, float* d_B,
                     float* d_C, int N);

                     