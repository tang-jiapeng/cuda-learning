include_directories(../common)

add_executable(elementwise main.cpp elementwise_add.cu)
target_link_libraries(elementwise PRIVATE ${CUDA_LIBRARIES})
target_compile_options(elementwise PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)