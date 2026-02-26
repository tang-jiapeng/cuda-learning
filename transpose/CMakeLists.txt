add_executable(transpose main.cpp transpose.cu)
target_include_directories(transpose PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(transpose PRIVATE common_gpu)
