cmake_minimum_required(VERSION 3.8)
project(gpugpu LANGUAGES CXX)

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
<<<<<<< HEAD
=======
#configure_file(include/cu/vector_add_kernel.cl ${CMAKE_BINARY_DIR}/vector_add_kernel.cl COPYONLY)

#if (BUILD_PYTHON)
#    add_library(cuckoo src/CuckooGPU/cuckoo.hpp)
#	target_link_libraries(cuckoo pybind11::module cuckoolib)
	#else()
	#add_library(cuckoo src/CuckooCPU/cuckoo.mm)
	#configure_file(src/CuckooCPU/MyKernels.metallib ${CMAKE_BINARY_DIR}/MyKernels.metallib COPYONLY)
	#endif()

#if (APPLE)
#    target_compile_definitions(cuckoo PRIVATE CUCKOO_SUPPORT_METAL=1)

#    find_library(METAL_LIBRARY Metal)
#    find_library(METALKIT_LIBRARY MetalKit)
#    find_library(COCOA_LIBRARY Cocoa)
#    find_library(CORE_FOUNDATION_LIBARY Foundation)

#    set(CMAKE_CXX_FLAGS "-x objective-c++ -fobjc-link-runtime")
#    xcrun metal -fcikernel src/CuckooCPU/example.metal -c -o src/CuckooCPU/MyLibrary.air
#    xcrun metallib -cikernel src/CuckooCPU/MyLibrary.air -o src/CuckooCPU/MyKernels.metallib

#    add_compile_options(cuckoo "-mmacosx-version-min=10.13,-fPIC,-ObjC++")
#    add_link_options(cuckoo "-fobjc-link-runtime,-mmacosx-version-min=10.13,-fPIC")
#
#    SET(EXTRA_LIBS "${METAL_LIBRARY}" "${METALKIT_LIBRARY}" "${COCOA_LIBRARY}" "${CORE_FOUNDATION_LIBARY}")
#    message(${EXTRA_LIBS})
#else()
#    set(EXTRA_LIBS "")
#endif()
>>>>>>> 084d6e60f26cd2db09e3b91d8e7d4e3be402d016
