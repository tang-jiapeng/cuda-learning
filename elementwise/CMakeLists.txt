add_executable(elementwise main.cpp elementwise_add.cu)
target_include_directories(elementwise PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CUDAToolkit_INCLUDE_DIRS}
)
