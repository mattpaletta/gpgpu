#pragma once
// DO NOT INCLUDE DIRECTLY, imported from `builder.hpp`
#ifndef BUILDER_CUDA_SAFE_IMPORT
#error "Do not import this file directly"
#endif

#define CHECK_CUDA(error) \
    do { \
        if (error != CUDA_SUCCESS) {    \
            const char* str;    \
            cuGetErrorName(error, &str);    \
            std::cout << "(CUDA) returned " << str; \
            std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ << "())" << std::endl;  \
            return false;   \
        }   \
    } while (0);

#define CHECK_CUDART(call) \
  do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
      std::cout << "(CUDART) returned " << cudaGetErrorString(status); \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl; \
      return false;  \
    } \
  } while (0)

#define CHECK_CUDART_THROW(call) \
  do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
      throw std::runtime_error("(CUDART) returned " + std::string(cudaGetErrorString(status)) + " (" + __FILE__ + ":" + std::to_string(__LINE__) + ":" + std::string(__func__) + "())");                                    \
    } \
  } while (0)
