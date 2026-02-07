cmake_minimum_required(VERSION 3.18)
project(CUDA_LEARNING)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O0 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-O0 ${CMAKE_CUDA_FLAGS}")
include(cmake/cuda.cmake)
set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(elementwise)