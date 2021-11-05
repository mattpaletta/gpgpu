#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <tuple>

#include "builder/line_comment.hpp"
#include "builder/function_arg.hpp"
#include "builder/kernel.hpp"
#include "runtime.hpp"

#ifdef GPGPU_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif // __APPLE__
#endif // GPGPU_OPENCL

#ifdef GPGPU_METAL
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#endif // GPGPU_METAL

#ifdef GPGPU_CUDA
#ifdef __WIN32__
#include "dbghelp.h"
#endif

// Macros defined in CMake
#include "jitify.hpp"

// Defines helper functions, only useful if CUDA
// Must be defined after 'jitify.hpp'
#define BUILDER_CUDA_SAFE_IMPORT
#include "builder+cuda.hpp"
#endif

namespace gpgpu {
	class Builder {
	private:
    	std::vector<std::unique_ptr<builder::Kernel>> funcs;
        const Runtime rt;

        std::string build_opencl() const;
        std::string build_cuda() const;
        std::string build_metal() const;

#ifdef GPGPU_METAL
        MTLResourceOptions GetResourceMode() const {
#if defined(TARGET_OS_IOS)
            return MTLResourceStorageModeShared;
#else
            return MTLResourceStorageModeManaged;
#endif
        }

        using MetalBuffer = id<MTLBuffer>;
        template<class RetT>
        MetalBuffer metal_add_parameter_ret_internal(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& argIndex, const std::size_t& dataCount) {
            const auto resourceMode = this->GetResourceMode();
            MetalBuffer outBuffer = [*device newBufferWithLength:sizeof(RetT) * dataCount options:resourceMode];
            [*commandEncoder setBuffer:outBuffer offset:0 atIndex:argIndex];
            return outBuffer;
        }

        // Process individual ARG
        template<class RetT, class A0>
        void metal_add_parameter_arg_internal(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data) {
            const auto resourceMode = this->GetResourceMode();
            auto inBuffer = [*device newBufferWithLength:sizeof(A0) * data.size() options:resourceMode];
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
        MetalBuffer metal_add_parameter_internal(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data) {
            // Process the last one
            this->metal_add_parameter_arg_internal<RetT, A0>(commandEncoder, device, argIndex, retDataCount, data);
            return this->metal_add_parameter_ret_internal<RetT>(commandEncoder, device, argIndex + 1, retDataCount); // Process the return value
        }

        template<class RetT, typename A0, typename... AN>
        MetalBuffer metal_add_parameter_internal(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data, AN... args) {
            this->metal_add_parameter_arg_internal<RetT, A0>(commandEncoder, device, argIndex, retDataCount, data); // Process A0
            return this->metal_add_parameter_internal<RetT>(commandEncoder, device, argIndex + 1, retDataCount, args...); // Process the rest ('recursively'), returns the 'return' buffer.
        }

        template<class RetT, class... AN>
        MetalBuffer metal_add_parameter(id<MTLComputeCommandEncoder>* commandEncoder, id<MTLDevice>* device, const std::size_t& retDataCount, AN... args) {
            return this->metal_add_parameter_internal<RetT>(commandEncoder, device, /* argIndex */ 0, retDataCount, args...);
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
            MetalBuffer outBuffer = this->metal_add_parameter<RetT>(&commandEncoder, &device, data.size(), data, args...);

            [commandEncoder setComputePipelineState:computePipelineState];
            [commandEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(data.size(), 1, 1)];
            [commandEncoder endEncoding];

            auto blitCommandEncoder = [commandBuffer blitCommandEncoder];
#if !defined(TARGET_OS_IOS)
            [blitCommandEncoder synchronizeResource:outBuffer];
#endif
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
        using OpenCLBuffer = cl::Buffer;
        
        // Process individual ARG
        template<typename A0>
        OpenCLBuffer* opencl_add_parameter_arg_internal(std::vector<std::unique_ptr<OpenCLBuffer>>* acc, cl::Kernel* func, cl::Context* context, cl::CommandQueue* queue, const bool should_write, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data) {
            auto buffer_N = std::make_unique<OpenCLBuffer>(*context, CL_MEM_READ_WRITE, sizeof(A0) * retDataCount);
            func->setArg(argIndex, *buffer_N.get());
            if (!should_write) {
                queue->enqueueWriteBuffer(*buffer_N.get(), CL_TRUE, 0, sizeof(A0) * retDataCount, data.data());
            }
            acc->emplace_back(std::move(buffer_N));
            return acc->back().get();
        }

        template<typename RetT>
        OpenCLBuffer* opencl_add_parameter_ret_internal(std::vector<std::unique_ptr<OpenCLBuffer>>* acc, cl::Kernel* func, cl::Context* context, cl::CommandQueue* queue, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<RetT>& data) {
            return this->opencl_add_parameter_arg_internal<RetT>(acc, func, context, queue, true /* Write */, argIndex, retDataCount, data);
        }

        // Process the last one.
        template<typename A0>
        OpenCLBuffer* opencl_add_parameter_internal(std::vector<std::unique_ptr<OpenCLBuffer>>* acc, cl::Kernel* func, cl::Context* context, cl::CommandQueue* queue, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data) {
            // Process the last one, which is the return value.
            return this->opencl_add_parameter_ret_internal<A0>(acc, func, context, queue, argIndex, retDataCount, data); // Process the return value
        }

        template<typename A0, typename... AN>
        OpenCLBuffer* opencl_add_parameter_internal(std::vector<std::unique_ptr<OpenCLBuffer>>* acc, cl::Kernel* func, cl::Context* context, cl::CommandQueue* queue, const std::size_t& argIndex, const std::size_t& retDataCount, const std::vector<A0>& data, const AN&... args) {
            this->opencl_add_parameter_arg_internal<A0>(acc, func, context, queue, false /* Read only */, argIndex, data.size(), data); // Process A0
            return this->opencl_add_parameter_internal(acc, func, context, queue, argIndex + 1, retDataCount, args...); // Process the rest ('recursively'), returns the 'return' buffer.
        }

        template<typename... AN>
        OpenCLBuffer* opencl_add_parameter(std::vector<std::unique_ptr<OpenCLBuffer>>* acc, cl::Kernel* func, cl::Context* context, cl::CommandQueue* queue, const std::size_t& retDataCount, const AN&... args) {
            return this->opencl_add_parameter_internal(acc, func, context, queue, /* argIndex */ 0, retDataCount, args...);
        }

        template<typename RetT, typename A0, typename ... AN>
        void run_opencl(const std::string& func_name, std::vector<RetT>* ret, const std::vector<A0>& data, const AN& ...args) {
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
            const std::string sourceCode = this->build_opencl();
            const auto a = std::make_pair(sourceCode.c_str(), sourceCode.size() + 1);
            cl::Program::Sources source(1, a);

            // Make program of the source code in the context
            auto program = cl::Program(context, source);
            const auto err_code = program.build(devices);
            if (err_code != 0) {
                std::cerr << "Failed to build kernel: " << std::endl;
                std::terminate();
            }

            // set up kernels and vectors for GPU code
            cl::Kernel func(program, func_name.c_str());
            
            std::vector<std::unique_ptr<OpenCLBuffer>> buffers; // RAII for OpenCL buffers.
            OpenCLBuffer* buffer_out = opencl_add_parameter(&buffers, &func, &context, &queue, data.size(), data, args..., *ret);
            std::cout << "Kernel Size: " << data.size() << ", 32" << std::endl;
            queue.enqueueNDRangeKernel(func, cl::NullRange, // kernel, offset
                cl::NDRange(data.size()),                   // global number of work items
                cl::NDRange(data.size() / 25));                           // local number (per group)
            
            ret->resize(data.size());
            queue.enqueueReadBuffer(*buffer_out, CL_TRUE, 0, sizeof(RetT) * data.size(), ret->data());
            queue.finish();
        }
#endif // GPGPU_OPENCL

#ifdef GPGPU_CUDA
        // Process individual ARG
        template<typename A0>
        A0* cuda_add_parameter_arg_internal(const std::size_t& retDataCount, const std::vector<A0>& h_data) {
            A0* d_data;
            CHECK_CUDART_THROW(cudaMalloc((void**) &d_data, sizeof(A0) * retDataCount));
            if (h_data.size() > 0) {
                CHECK_CUDART_THROW(cudaMemcpy(d_data, h_data.data(), sizeof(A0) * retDataCount, cudaMemcpyHostToDevice));
            }
            return d_data;
        }

        template<typename A0, typename... AN>
        std::tuple<A0*, AN*...> cuda_add_parameter_internal(const std::size_t& retDataCount, const std::vector<A0>& data, const std::vector<AN>&... args) {
            return { this->cuda_add_parameter_arg_internal<A0>(data.size(), data), this->cuda_add_parameter_arg_internal(retDataCount, args...) };
        }

        template<typename... AN>
        std::tuple<AN*...> cuda_add_parameter(const std::size_t& retDataCount, const std::vector<AN>&... args) {
            return this->cuda_add_parameter_internal(retDataCount, args...);
        }

        template<typename A0>
        bool cuda_free_parameter(A0* deviceData) {
            CHECK_CUDART(cudaFree(deviceData));
            return true;
        }

        template<typename A0, typename... AN>
        bool cuda_free_parameter(A0* data, AN*... args) {
            return this->cuda_free_parameter<A0>(data) && this->cuda_free_parameter<AN...>(args...);
        }

        // Start Call Utility Function
        template<typename Function, typename Tuple, size_t ... I>
        auto call(Function f, Tuple t, std::index_sequence<I ...>) { 
            return f(std::get<I>(t) ...); 
        }
        template<typename Function, typename Tuple>
        auto call(Function f, Tuple t) {
            static constexpr auto size = std::tuple_size<Tuple>::value;
            return call(f, t, std::make_index_sequence<size>{});
        }
        // End Call Utility Function

        template<typename RetT, typename A0, typename ... AN>
        bool run_cuda(const std::string& func_name, std::vector<RetT>* ret, const std::vector<A0>& data, const std::vector<AN>& ...args) {
            const std::string sourceCode = this->build_cuda();
            thread_local static jitify::JitCache kernel_cache;
            jitify::Program program = kernel_cache.program(func_name+"\n" + sourceCode, 0, { "-std=c++14",});
            std::tuple<A0*, AN*...> deviceData = this->cuda_add_parameter(data.size(), data, args...);
            RetT* returnDeviceData = this->cuda_add_parameter_arg_internal(data.size(), *ret);

            //using jitify::reflection::type_of;
            // constexpr auto sizeTuple = std::tuple_size<decltype(deviceData)>::value;
            CHECK_CUDA(this->call([&program, &func_name, &data, returnDeviceData](auto&... args) -> CUresult {
                dim3 grid(1);
                dim3 block(data.size());
                return program.kernel(func_name)
                    .instantiate()
                    .configure(grid, block)
                    .launch(args..., returnDeviceData);
                }, deviceData));

            // The last item is the return value, copy it to the 'ret' parameter
            ret->resize(data.size());
            CHECK_CUDART(cudaMemcpy(ret->data(), returnDeviceData, sizeof(RetT) * data.size(), cudaMemcpyDeviceToHost));
            return this->call([this](auto... args) -> bool {
                return this->cuda_free_parameter(args...);
            }, deviceData) && this->cuda_free_parameter(returnDeviceData);
        }
#endif // GPGPU_CUDA

        bool hasMetalDevice() const;
        bool hasOpenCLDevice() const;
        bool hasCudaDevice() const;
	public:
        Builder(const Runtime& _rt);
        ~Builder();

        builder::Kernel* GetKernel(const std::string& name);
        builder::Kernel* NewKernel(const std::string& name, std::vector<std::unique_ptr<builder::FunctionArg>>&& args, const std::string& returnType);

        std::string dump(const Runtime& rt);
        std::string dump();

        bool hasDevice(const Runtime& rt) const;
        bool hasDevice() const;

        template<typename RetT, typename... AN>
        void run(const Runtime& rt, const std::string& func_name, std::vector<RetT>* ret, const AN&... args) {
            switch (rt) {
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
#ifdef GPGPU_CUDA
            case CUDA:
                if (!this->run_cuda<RetT>(func_name, ret, args...)) {
                    throw new std::runtime_error("CUDA ERROR");
                }
                break;
#endif
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
