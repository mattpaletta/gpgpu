//
// Created by mattp on 7/24/2019.
//

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

#include <cassert>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

int main() {
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

        cl::Program::Sources sources;

        // Read source file
        const std::string sourceFileLocation = "vector_add_kernel.cl";
        std::ifstream sourceFile(sourceFileLocation);
        std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));
        const auto a = std::make_pair(sourceCode.c_str(), sourceCode.size() + 1);
        cl::Program::Sources source(1, a);

        // Make program of the source code in the context
        auto program = cl::Program(context, source);

        // Build program for these specific devices
        const auto err_code = program.build(devices);
        if (err_code != 0) {
            std::cerr << "Failed to build kernel: " << std::endl;
            std::terminate();
        }
        std::cout << "Building kernel:" << sourceFileLocation << std::endl;
        std::cout << sourceCode.c_str() << std::endl;

        // Output the kernel(s) to resource file.
        std::ofstream out("resources.hpp");
        out << "#pragma once" << std::endl;
        out << "#include <string>" << std::endl;
        out << "const std::string vector_add_kernel_cl = \"";
        std::istringstream iss(sourceCode);
        for (std::string line; std::getline(iss, line); ) {
            // TODO: Remove comments.
            out << line;
        }
        out << "\";";
        out.close();

    } catch(cl::Error& error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        return 1;
    }

    return 0;
}