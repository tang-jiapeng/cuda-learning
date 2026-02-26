add_executable(reduce main.cpp reduce.cu)
target_include_directories(reduce PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(reduce PRIVATE common_gpu)
