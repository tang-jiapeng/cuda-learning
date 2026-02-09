include_directories(../common)

add_executable(elementwise main.cpp elementwise_add.cu)
target_include_directories(elementwise PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
# target_compile_options(elementwise PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
# )