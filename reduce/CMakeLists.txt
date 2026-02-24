include_directories(../common)

add_executable(reduce main.cpp reduce.cu)
target_include_directories(reduce PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
