//
// Created by mattp on 7/24/2019.
//

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif


#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Create the two input vectors
    constexpr int N = 1'000'000;
    int *A = new int[N];
    int *B = new int[N];

    for(int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = N - i;
    }

    try {
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

        // Create a command queue and use the first device
        cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);


        std::cout << "Got devices" << std::endl;

        cl::Program::Sources sources;

        // Read source file
        std::ifstream sourceFile("vector_add_kernel.cl");
        std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Make program of the source code in the context
        auto program = cl::Program(context, source);

        // Build program for these specific devices
        program.build(devices);
        std::cout << "Building kernel" << std::endl;

        // Make kernel
        cl::Kernel kernel(program, "vector_add");

        // Create memory buffers
        auto bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, N * sizeof(int));
        auto bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, N * sizeof(int));
        auto bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, N * sizeof(int));

        // Copy lists A and B to the memory buffers
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, N * sizeof(int), A);
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, N * sizeof(int), B);

        // Set arguments to kernel
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);

        // Run the kernel on specific ND range
        cl::NDRange global(N);
        cl::NDRange local(1);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

        // Read buffer C into a local list
        int *C = new int[N];
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N * sizeof(int), C);

//        for(int i = 0; i < N; ++i) {
//            std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
//        }

        std::cout << "Finished processing" << std::endl;

    } catch(cl::Error& error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

    return 0;
}