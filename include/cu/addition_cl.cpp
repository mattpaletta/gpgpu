//
// Created by mattp on 7/24/2019.
//

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

#include <numeric>
#include <functional>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

#include <cassert>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>

// #include <avalanche/terminal_nodes.h>
// #include <avalanche/nodes.h>
//#include <avalanche/Shape.h>
// #include <avalanche/Executor.h>

int main() {
	/*
    // Create the two input vectors
    constexpr int N = 1'000;
    constexpr int full_size = N * N;
    std::vector<float> data_vector(full_size);
    for (int n = 0; n < full_size; ++n) {
        data_vector[n] = n;
    }
    auto val1 = avalanche::Constant::tensor<float>(
                                        data_vector,
                                        avalanche::Shape({N, N}));
    auto val2 = avalanche::Constant::tensor<float>(
                                        data_vector,
                                        avalanche::Shape({N, N}));

    // Get available platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(),
        0
    };
    cl::Context context( CL_DEVICE_TYPE_GPU, cps);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Found devices: " << devices.size() << std::endl;

    auto output = avalanche::F<avalanche::MatMul>(val1, val2, false, false);
    avalanche::Executor executor(avalanche::Context::make_for_device(0), {output});
    auto start_time = std::chrono::steady_clock::now();
    for (int timeit = 0; timeit < 10; ++timeit) {
        const auto results = executor.run();
        std::vector<float> cpu_copy;
        results[0]->fetch_data_into(cpu_copy);
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Time elapsed" << diff.count() << " s\n";
    return 0;
	*/
}
