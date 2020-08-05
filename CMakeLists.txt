cmake_minimum_required(VERSION 3.8)
project(gpugpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/fetch_extern.cmake)

option(GPGPU_USE_CUDA "Enable CUDA support" OFF)
option(GPGPU_USE_OPENCL "Enable OpenCL support" ON)
option(GPGPU_USE_METAL "Enable Metal support" ON)

if (${GPGPU_USE_OPENCL})
	fetch_extern(avalanche https://github.com/mattpaletta/avalanche.git master)
endif()
add_executable(addition_cpu include/cu/addition_cpu.cpp)

if (${GPGPU_USE_CUDA})
	FIND_PACKAGE(CUDA REQUIRED)
	cuda_add_executable(addition_cuda include/cu/addition_cuda.cu)
endif()

if (${GPGPU_USE_OPENCL})
	add_executable(addition_cl include/cu/addition_cl.cpp)
	target_link_libraries(addition_cl avalanche)
	set_target_properties(addition_cl PROPERTIES SKIP_BUILD_RPATH ON)
endif()

if (${GPGPU_USE_METAL})
	find_library(METAL_LIBRARY Metal)
	find_library(METALKIT_LIBRARY MetalKit)
	find_library(METALPERFORMANCE_LIBRARY MetalPerformanceShaders)
	find_library(COCOA_LIBRARY Cocoa)
	find_library(CORE_FOUNDATION_LIBARY Foundation)

	add_executable(addition_metal include/cu/addition_metal.mm)
	target_link_libraries(addition_metal ${METAL_LIBRARY} ${METALKIT_LIBRARY} ${METALPERFORMANCE_LIBRARY} ${COCOA_LIBRARY} ${CORE_FOUNDATION_LIBARY})
endif()

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
