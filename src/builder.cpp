#include <gpgpu/builder.hpp>

using namespace gpgpu;

Builder::Builder(const Runtime& _rt) : rt(_rt) {};
Builder::~Builder() = default;

std::string Builder::build_opencl() const {
    const std::string imports = "";
    std::string final_kernel = imports;
    for (const auto& kern : this->funcs) {
        final_kernel += kern->build_opencl() + "\n\n";
    }
    return final_kernel;
}

std::string Builder::build_cuda() const {
    const std::string imports = "";
    std::string final_kernel = imports;
    for (const auto& kern : this->funcs) {
        final_kernel += kern->build_cuda() + "\n\n";
    }
    return final_kernel;
}

std::string Builder::build_metal() const {
    const std::string imports = "#include <metal_stdlib>\nusing namespace metal;\n\n";
    std::string final_kernel = imports;
    for (const auto& kern : this->funcs) {
        final_kernel += kern->build_metal() + "\n\n";
    }
    return final_kernel;
}

builder::Kernel* Builder::GetKernel(const std::string& name) {
    for (auto& f : this->funcs) {
        if (f->getName() == name) {
            return f.get();
        }
    }
    return nullptr;
}

builder::Kernel* Builder::NewKernel(const std::string& name, std::vector<std::unique_ptr<builder::FunctionArg>>&& args, const std::string& returnType) {
    this->funcs.emplace_back(std::move(std::make_unique<builder::Kernel>(name, std::move(args), returnType)));
    return this->funcs.back().get();
}

bool Builder::hasMetalDevice() const {
#ifdef GPGPU_METAL
    return true;
#else
    return false;
#endif
}

bool Builder::hasOpenCLDevice() const {
#ifdef GPGPU_OPENCL
    return true;
#else
    return false;
#endif
}

bool Builder::hasCudaDevice() const {
#ifdef GPGPU_CUDA
    int count;
    auto result = cudaGetDeviceCount(&count);
    if (result == cudaSuccess) {
        return count > 0;
    } else {
        return false;
    }
#else
    return false;
#endif
}

bool Builder::hasDevice() const {
    return this->hasDevice(this->rt);
}

bool Builder::hasDevice(const Runtime& rt) const {
    switch (rt) {
    case CUDA:
        return this->hasCudaDevice();
    case OpenCL:
        return this->hasOpenCLDevice();
    case Metal:
        return this->hasMetalDevice();
    default:
        return false;
    }
}

std::string Builder::dump(const Runtime& rt) {
    switch (rt) {
    case OpenCL:
        return this->build_opencl();
    case Metal:
        return this->build_metal();
    case CUDA:
        return this->build_cuda();
        //                case CPU:
        //                    return this->build_cpu();
    }
    return "";
}

std::string Builder::dump() {
    return this->dump(this->rt);
}