include_directories(../common)

add_executable(transpose main.cpp transpose.cu cublas_transpose.cu)
target_include_directories(transpose PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(transpose PRIVATE CUDA::cublas)
