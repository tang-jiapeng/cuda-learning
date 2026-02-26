add_executable(gemm main.cpp gemm.cu)
target_include_directories(gemm PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gemm PRIVATE common_gpu)
