include_directories(../common)

add_executable(gemm main.cpp gemm.cu)
target_include_directories(gemm PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
