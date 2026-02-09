cmake_minimum_required(VERSION 3.18)
project(CUDA_LEARNING)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(cmake/cuda.cmake)

# --- 设置默认构建类型 (如果没有在命令行指定) ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release) # 默认使用 Release，跑分最快
endif()

# --- 针对 C++ Host 代码的设置 ---
# Debug: 关优化(-O0), 生成调试信息(-g)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
# Release: 最高优化(-O3)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# --- 针对 CUDA Device 代码的设置 ---
# Debug: 
# -g: Host端调试信息
# -G: Device端调试信息 (严重影响性能，但允许在 Nsight/cuda-gdb 中调试 Kernel)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Release: 
# -O3: 最高优化
# --use_fast_math: 可选，开启快速数学运算（牺牲一点精度换速度，适合跑分）
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# 打印当前模式
message(STATUS "Current Build Type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(elementwise)