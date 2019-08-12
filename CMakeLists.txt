cmake_minimum_required(VERSION 3.8)

project(gpugpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(addition_cpu include/cu/addition_cpu.cpp)


FIND_PACKAGE(CUDA)
cuda_add_executable(addition_cuda include/cu/addition_cuda.cu)

FIND_PACKAGE(OpenCL)
add_executable(addition_cl include/cu/addition_cl.cpp include/cu/vector_add_kernel.cl)
target_link_libraries(addition_cl OpenCL::OpenCL)
#configure_file(include/cu/vector_add_kernel.cl ${CMAKE_BINARY_DIR}/vector_add_kernel.cl COPYONLY)