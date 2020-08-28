#include <gpgpu/builder/function_arg.hpp>
using namespace gpgpu::builder;

FunctionArg::FunctionArg(const ARG_TYPE& _location, const std::string& _type, const std::string& _name, const bool _is_const) : BaseBuilder(), location(_location), type(_type), is_const(_is_const), name(_name) {}

std::string FunctionArg::opencl_location_to_string() const {
    switch (this->location) {
    case GLOBAL:
        return "global";
    case DEVICE:
        return "device";
    case HOST:
        return "host";
    }
    return "";
}

std::string FunctionArg::metal_location_to_string() const {
    switch (this->location) {
    case GLOBAL:
        // Metal copies to the device
        return "device";
    case DEVICE:
        return "device";
    case HOST:
        return "host";
    }
    return "";
}

std::string FunctionArg::cuda_location_to_string() const {
    switch (this->location) {
    case GLOBAL:
        return "__global__";
    case DEVICE:
        return "__device__";
    case HOST:
        return "__host__";
    }
    return "";
}

std::string FunctionArg::constStr() const {
    return this->is_const ? "const " : "";
}

std::string FunctionArg::build_opencl_arg(const std::size_t & indentation, const std::size_t & i) const {
    return this->getIndentation(0) + this->opencl_location_to_string() + " " + this->constStr() + this->type + " " + name + (this->val ? " = " + this->val->build_opencl(0) : "");
}

std::string FunctionArg::build_metal_arg(const std::size_t & indentation, const std::size_t & i) const {
    return this->getIndentation(0) + this->constStr() + this->metal_location_to_string() + " " + this->type + " " + name + +"[[ buffer(" + std::to_string(i) + ") ]]" + (this->val ? " = " + this->val->build_metal(0) : "");
}

std::string FunctionArg::build_cuda_arg(const std::size_t & indentation, const std::size_t & i) const {
    return this->getIndentation(0) + /*this->constStr() + */ this->type + " " + name + (this->val ? " = " + this->val->build_cuda(0) : "");
}

std::string FunctionArg::build_cpu_arg(const std::size_t & indentation, const std::size_t & i) const {
    return ""; 
}