cmake_minimum_required(VERSION 3.8)
project(gpugpu LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/fetch_extern.cmake)


if (APPLE)
	# OpenCL Implementation doesn't seem to work on apple, only build metal
	option(GPGPU_USE_METAL "Enable Metal support" ON)
	if (${GPGPU_USE_METAL})
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -x objective-c++")
	endif()
else()
	option(GPGPU_USE_OPENCL "Enable OpenCL support" ON)
	find_package(CUDA QUIET)
	if (CUDA_FOUND)
		option(GPGPU_USE_CUDA "Enable CUDA support" ON)
	endif()
endif()

file(GLOB_RECURSE INC_FILES include/gpgpu/*.hpp)
file(GLOB_RECURSE SRC_FILES src/*.cpp)

add_executable(addition_cpu include/cu/addition_cpu.cpp)
if (${GPGPU_USE_METAL})
	set_target_properties(addition_cpu PROPERTIES LINKER_LANGUAGE OBJCXX)
endif()

add_library(gpgpu STATIC ${INC_FILES} ${SRC_FILES})
target_include_directories(gpgpu PUBLIC include)

set(GPGPU_DEPS)

if (${GPGPU_USE_CUDA})
	FIND_PACKAGE(CUDA REQUIRED)
	find_library(cuda REQUIRED)

	message("Cuda: ${CUDA_LIBRARIES}")

	fetch_extern(jitify https://github.com/NVIDIA/jitify.git b4f05a146121085805ad1cbe64537933756d1a17)
	add_library(jitify INTERFACE)
	target_include_directories(jitify INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/_deps/jitify-src)
	# target_link_libraries(jitify INTERFACE cuda cudart nvrtcc)

	list(APPEND GPGPU_DEPS ${CUDA_LIBRARIES} jitify)
	target_compile_definitions(gpgpu PUBLIC "-DGPGPU_CUDA")

	cuda_add_executable(addition_cuda include/cu/addition_cuda.cu)
endif()

if (${GPGPU_USE_OPENCL})
	find_package(OpenCL REQUIRED)
	list(APPEND GPGPU_DEPS OpenCL::OpenCL)
	target_compile_definitions(gpgpu PUBLIC "-DGPGPU_OPENCL")

	add_executable(addition_cl include/cu/addition_cl.cpp)
	if (${GPGPU_USE_METAL})
		set_target_properties(addition_cl PROPERTIES LINKER_LANGUAGE OBJCXX)
	endif()
	message("Using OpenCL")
endif()

if (${GPGPU_USE_METAL})
	find_library(METAL_LIBRARY Metal)
	find_library(METALKIT_LIBRARY MetalKit)
	find_library(METALPERFORMANCE_LIBRARY MetalPerformanceShaders)
	find_library(COCOA_LIBRARY Cocoa)
	find_library(CORE_FOUNDATION_LIBARY Foundation)

	message("Using Metal")

	set(METAL_LIBS ${METAL_LIBRARY} ${METALKIT_LIBRARY} ${METALPERFORMANCE_LIBRARY} ${COCOA_LIBRARY} ${CORE_FOUNDATION_LIBARY})

	add_executable(addition_metal include/cu/addition_metal.mm)
	target_link_libraries(addition_metal ${METAL_LIBS})

	list(APPEND GPGPU_DEPS ${METAL_LIBS})
	target_compile_definitions(gpgpu PUBLIC "-DGPGPU_METAL")
endif()

target_link_libraries(gpgpu PUBLIC ${GPGPU_DEPS})
if (${GPGPU_USE_METAL})
	set_target_properties(gpgpu PROPERTIES LINKER_LANGUAGE OBJCXX)
endif()

# Build tests
set(CATCH_VERSION 0f12995501ee01d3d2bdd9f4978bb28b5f670bab)
if (NOT TARGET Catch2::Catch2)
	fetch_extern(catch2 https://github.com/catchorg/Catch2 ${CATCH_VERSION})
endif()
add_executable(test_gpgpu test/test_gpgpu.cpp)
target_link_libraries(test_gpgpu gpgpu Catch2::Catch2)
if (${GPGPU_USE_METAL})
	set_target_properties(test_gpgpu PROPERTIES LINKER_LANGUAGE OBJCXX)
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
