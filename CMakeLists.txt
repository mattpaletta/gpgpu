cmake_minimum_required(VERSION 3.8)

project(gpugpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(addition_cpu include/cu/addition_cpu.cpp)

FIND_PACKAGE(CUDA)
cuda_add_executable(addition_cuda include/cu/addition_cuda.cu)

FIND_PACKAGE(OpenCL)
# # This program compiles the opencl resources, and creates a 'resources.hpp' file.
add_executable(addition_cl_res include/compile_cl.cpp include/cu/vector_add_kernel.cl)
target_link_libraries(addition_cl_res OpenCL::OpenCL)

add_custom_target(create_resources
                  COMMAND ./addition_cl_res
                  DEPENDS addition_cl_res
                  BYPRODUCTS ${CMAKE_BINARY_DIR}/resources.hpp)
configure_file(include/cu/vector_add_kernel.cl ${CMAKE_BINARY_DIR}/vector_add_kernel.cl COPYONLY)

add_executable(addition_cl include/cu/addition_cl.cpp ${CMAKE_BINARY_DIR}/resources.hpp)
target_include_directories(addition_cl PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(addition_cl OpenCL::OpenCL)
