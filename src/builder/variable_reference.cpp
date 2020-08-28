#include <gpgpu/builder/variable_reference.hpp>

using namespace gpgpu::builder;

std::string VariableReference::build_shared(const std::size_t& indentation) const {
    return this->getIndentation(indentation) + this->var;
}

VariableReference::VariableReference(const std::string& _var) : BaseBuilder(), var(_var) {}

std::string VariableReference::build_opencl(const std::size_t& indentation) const {
    return this->build_shared(indentation); 
}

std::string VariableReference::build_metal(const std::size_t& indentation) const {
    return this->build_shared(indentation); 
}

std::string VariableReference::build_cuda(const std::size_t& indentation) const {
    return this->build_shared(indentation); 
}

std::string VariableReference::build_cpu(const std::size_t& indentation) const {
    return ""; 
}