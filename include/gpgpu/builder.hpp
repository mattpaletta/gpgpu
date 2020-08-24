#pragma once
#include <string>
#include <vector>
#include <iostream>

#include "gpgpu/builder/line_comment.hpp"
#include "gpgpu/builder/function_arg.hpp"
#include "gpgpu/builder/kernel.hpp"
#include "gpgpu/runtime.hpp"

#ifdef GPGPU_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif // __APPLE__
#endif // GPGPU_METAL

#ifdef GPGPU_METAL
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#endif

namespace gpgpu {
	class Builder {
	private:
		std::vector<std::unique_ptr<builder::Kernel>> funcs;
		const Runtime rt;

        std::string build_opencl() const {
            const std::string imports = "";
            std::string final_kernel = imports;
            for (const auto& kern : this->funcs) {
                final_kernel += kern->build_opencl() + "\n\n";
            }

            return final_kernel;
        }

        std::string build_cuda() const {
            const std::string imports = "";
            std::string final_kernel = imports;
            for (const auto& kern : this->funcs) {
                final_kernel += kern->build_cuda() + "\n\n";
            }

            return final_kernel;
        }

        std::string build_metal() const {
            const std::string imports = "#include <metal_stdlib>\nusing namespace metal;\n\n";
            std::string final_kernel = imports;
            for (const auto& kern : this->funcs) {
                final_kernel += kern->build_metal() + "\n\n";
            }

            return final_kernel;
        }

//        std::string build_cpu() const {
//            const std::string imports = "";
//            std::string final_kernel = imports;
//            for (const auto& kern : this->funcs) {
//                final_kernel += kern->build_cpu() + "\n\n";
//            }
//
//            return final_kernel;
//        }

#ifdef GPGPU_METAL
        using MetalBuffer = id<MTLBuffer>;
        template<class RetT>
        MetalBuffer add_parameter_ret_internal(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& argIndex, const std::size_t& dataCount) {
            MetalBuffer outBuffer = [*device newBufferWithLength:sizeof(RetT) * dataCount options:MTLResourceStorageModeManaged];
            [*commandEncoder setBuffer:outBuffer offset:0 atIndex:argIndex];
            return outBuffer;
        }

        // Process individual ARG
        template<class RetT, class A0>
        void add_parameter_arg_internal(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data) {
            auto inBuffer = [*device newBufferWithLength:sizeof(A0) * data.size() options:MTLResourceStorageModeManaged];
            auto* inData = static_cast<A0*>(inBuffer.contents);
            for (std::size_t i = 0; i < data.size(); ++i) {
                // update input data
                inData[i] = data.at(i);
            }
            [inBuffer didModifyRange: NSMakeRange(0, sizeof(A0) * data.size())];
            [*commandEncoder setBuffer:inBuffer offset:0 atIndex:argIndex];
        }

        // Process the last one.
        template<class RetT, class A0>
        MetalBuffer add_parameter_internal(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data) {
            // Process the last one
            this->add_parameter_arg_internal<RetT, A0>(commandEncoder, device, argIndex, retDataCount, data);
            return this->add_parameter_ret_internal<RetT>(commandEncoder, device, argIndex + 1, retDataCount); // Process the return value
        }

        template<class RetT, typename A0, typename... AN>
        MetalBuffer add_parameter_internal(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data, AN... args) {
            this->add_parameter_arg_internal<RetT, A0>(commandEncoder, device, argIndex, retDataCount, data); // Process A0
            return this->add_parameter_internal<RetT>(commandEncoder, device, argIndex + 1, retDataCount, args...); // Process the rest ('recursively'), returns the 'return' buffer.
        }

        template<class RetT, class... AN>
        MetalBuffer add_parameter(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& retDataCount, AN... args) {
            return this->add_parameter_internal<RetT>(commandEncoder, device, /* argIndex */ 0, retDataCount, args...);
        }

        template<class RetT, class A0, class ... AN>
        void run_metal(const std::string& func_name, std::vector<RetT>* ret, const std::vector<A0>& data, AN ...args) {
            const std::string shader = this->build_metal();
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();

            NSString* funcNameObjC = [NSString stringWithCString:func_name.c_str() encoding:NSUTF8StringEncoding];
            NSString* shadersSrc = [NSString stringWithCString:shader.c_str() encoding:NSUTF8StringEncoding];
            auto library = [device newLibraryWithSource:shadersSrc options:[MTLCompileOptions alloc] error: nullptr];
            auto entryFunc = [library newFunctionWithName:funcNameObjC];
            assert(entryFunc != nullptr);

            auto computePipelineState = [device newComputePipelineStateWithFunction:entryFunc error:nullptr];
            auto commandQueue = [device newCommandQueue];

            auto commandBuffer = [commandQueue commandBuffer];
            auto commandEncoder = [commandBuffer computeCommandEncoder];

            // Parameters
            MetalBuffer outBuffer = this->add_parameter<RetT>(&commandEncoder, &device, data.size(), data, args...);

            [commandEncoder setComputePipelineState:computePipelineState];
            [commandEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(data.size(), 1, 1)];
            [commandEncoder endEncoding];

            auto blitCommandEncoder = [commandBuffer blitCommandEncoder];
            [blitCommandEncoder synchronizeResource:outBuffer];
            [blitCommandEncoder endEncoding];

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            auto* outData = static_cast<RetT*>(outBuffer.contents);
            ret->resize(data.size());
            for (std::size_t i = 0; i < data.size(); ++i) {
                // Copy to ret vector.
                ret->at(i) = outData[i];
            }
        }
#endif // GPGPU_METAL

#ifdef GPGPU_OPENCL
        template<class RetT, class A0, class ... AN>
        void run_opencl(const std::string& func_name, std::vector<RetT>* ret, const std::vector<A0>& data, AN ...args) {
            // Get available platforms
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);

            // Select the default platform and create a context using this platform and the GPU
            cl_context_properties cps[3] = {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)(platforms[0])(),
                0
            };
            cl::Context context(CL_DEVICE_TYPE_GPU, cps);

            // Get a list of devices on this platform
            std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

            // Create a command queue and use the first device
            cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

            cl::Program::Sources sources;
            std::string sourceCode = this->build_opencl();
            const auto a = std::make_pair(sourceCode.c_str(), sourceCode.size() + 1);
            cl::Program::Sources source(1, a);

            // Make program of the source code in the context
            auto program = cl::Program(context, source);
            const auto err_code = program.build(devices);
            if (err_code != 0) {
                std::cerr << "Failed to build kernel: " << std::endl;
                std::terminate();
            }


        }
#endif // GPGPU_OPENCL

	public:
		Builder(const Runtime& _rt) : rt(_rt) {};
		~Builder() = default;

		builder::Kernel* GetKernel(const std::string& name) {
			for (auto& f : this->funcs) {
				if (f->getName() == name) {
                    return f.get();
				}
			}
			return nullptr;
		}

		builder::Kernel* NewKernel(const std::string& name, std::vector<std::unique_ptr<builder::FunctionArg>>&& args, const std::string& returnType) {
			this->funcs.emplace_back(std::make_unique<builder::Kernel>(name, std::move(args), returnType));
            return this->funcs.back().get();
		}

        std::string dump() {
            switch(rt) {
                case OpenCL:
                    return this->build_opencl();
                case Metal:
                    return this->build_metal();
                case CUDA:
                    return this->build_cuda();
//                case CPU:
//                    return this->build_cpu();
            }
        }

        template<typename RetT, typename... AN>
        void run(const Runtime& rt, const std::string& func_name, std::vector<RetT>* ret, AN... args) {
            switch(this->rt) {
#ifdef GPGPU_OPENCL
                case OpenCL:
                    this->run_opencl<RetT>(func_name, ret, args...);
                    break;
#endif
#ifdef GPGPU_METAL
                case Metal:
                    this->run_metal<RetT>(func_name, ret, args...);
                    break;
#endif
                    //                case CUDA:
                    //                    return this->build_cuda();
                default:
                    throw new std::runtime_error("Unsupported backend");
            }
        }

        template<typename RetT, typename... AN>
        void run(const std::string& func_name, std::vector<RetT>* ret, AN... args) {
            this->run(this->rt, func_name, ret, std::forward<AN>(args)...);
        }
	};
}
