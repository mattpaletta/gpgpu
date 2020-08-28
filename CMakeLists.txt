cmake_minimum_required(VERSION 3.17)
project(gpugpu LANGUAGES CXX)
if (APPLE)
	enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(cmake/fetch_extern.cmake)

if (APPLE)
	# OpenCL Implementation doesn't seem to work on apple, only build metal
	option(GPGPU_USE_METAL "Enable Metal support" ON)
	if (${GPGPU_USE_METAL})
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -x objective-c++")
	endif()
else()
	option(GPGPU_USE_OPENCL "Enable OpenCL support" ON)
	set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
	set(CUDA_SEPARABLE_COMPILATION ON)
	find_package(CUDA QUIET)
	if (CUDA_FOUND)
		option(GPGPU_USE_CUDA "Enable CUDA support" ON)
		enable_language(CUDA)
	else()
		option(GPGPU_USE_CUDA "Enable CUDA support" OFF)
	endif()
endif()

file(GLOB_RECURSE INC_FILES include/gpgpu/*.hpp)
file(GLOB_RECURSE SRC_FILES src/*.cpp)

add_executable(addition_cpu include/cu/addition_cpu.cpp)
if (${GPGPU_USE_METAL})
	set_target_properties(addition_cpu PROPERTIES LINKER_LANGUAGE OBJCXX)
endif()

if (${GPGPU_USE_CUDA})
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

	set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
	set(CUDA_SEPARABLE_COMPILATION ON)
	find_package(CUDA REQUIRED)
	cuda_add_library(gpgpu STATIC ${INC_FILES} ${SRC_FILES})
else()
	add_library(gpgpu STATIC ${INC_FILES} ${SRC_FILES})
endif()

target_include_directories(gpgpu PUBLIC include)
set(GPGPU_DEPS)

if (${GPGPU_USE_CUDA})
	message("Cuda: ${CUDA_LIBRARIES}")
	if (WIN32)
		find_library(nvrtc REQUIRED)
		find_library(cuda REQUIRED)
		set(CUDA_NVRTC nvrtc)
		set(CUDA_CUDA cuda)
	else()
		find_library(CUDA_NVRTC nvrtc PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 REQUIRED)
		# find_library(CUDA_CUDA cuda PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 QUIET)
	endif()
	if (WIN32)
		find_library(dbghelp REQUIRED)
		set(DBGHELP_LIB dbghelp)
	endif()
	fetch_extern(jitify https://github.com/NVIDIA/jitify.git b4f05a146121085805ad1cbe64537933756d1a17)
	add_library(jitify INTERFACE)
	target_include_directories(jitify INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/_deps/jitify-src)
	target_link_libraries(jitify INTERFACE ${CUDA_CUDA} ${CUDA_LIBRARIES} ${CUDA_NVRTC} ${CMAKE_DL_LIBS} ${DBGHELP_LIB})

	message("Jitify: ${CUDA_CUDA} ${CUDA_LIBRARIES} ${CUDA_NVRTC} ${CMAKE_DL_LIBS} ${DBGHELP_LIB}")

	# If not msvc, add cross-execution-space-call. This is blocked under msvc by a jitify related bug (untested > CUDA 10.1): https://github.com/NVIDIA/jitify/issues/62
    #if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    #    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} cross-execution-space-call")
    #endif()

    # If not msvc, add reorder to Werror. This is blocked under msvc by cub/thrust and the lack of isystem on msvc. Appears unable to suppress the warning via diag_suppress pragmas.
    #if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    #    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} reorder")
    #endif()

	list(APPEND GPGPU_DEPS jitify)
	target_compile_definitions(gpgpu PUBLIC "-DGPGPU_CUDA")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DJITIFY_PRINT_ALL")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJITIFY_PRINT_ALL")

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

message("GPGPU Libraries: ${GPGPU_DEPS}")
target_link_libraries(gpgpu PUBLIC ${GPGPU_DEPS})
if (${GPGPU_USE_METAL})
	set_target_properties(gpgpu PROPERTIES LINKER_LANGUAGE OBJCXX)
endif()

# Build tests
set(CATCH_VERSION 0f12995501ee01d3d2bdd9f4978bb28b5f670bab)
if (NOT TARGET Catch2::Catch2)
	fetch_extern(catch2 https://github.com/catchorg/Catch2 ${CATCH_VERSION})
endif()

if (${GPGPU_USE_CUDA})
	cuda_add_executable(test_gpgpu test/test_gpgpu.cpp)
	set_source_files_properties(test/test_gpgpu.cpp PROPERTIES LANGUAGE CUDA)
else()
	add_executable(test_gpgpu test/test_gpgpu.cpp)
endif()

target_link_libraries(test_gpgpu PUBLIC gpgpu Catch2::Catch2)
if (GPGPU_USE_METAL AND NOT GPGPU_USE_CUDA)
	set_target_properties(test_gpgpu PROPERTIES LINKER_LANGUAGE OBJCXX)
elseif(GPGPU_USE_METAL AND GPGPU_USE_CUDA)
	message(FATAL "Cannot use CUDA and Metal at the same time.")
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
