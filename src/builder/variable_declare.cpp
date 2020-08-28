#include <gpgpu/builder/variable_declare.hpp>
using namespace gpgpu::builder;

VariableDeclare::VariableDeclare(const std::string& _type, const std::string& _name, std::unique_ptr<BaseBuilder>&& _val) : BaseBuilder(), type(_type), name(_name), val(std::move(_val)) {}
VariableDeclare::VariableDeclare(const std::string& _type, const std::string& _name) : BaseBuilder(), type(_type), name(_name), val() {}

std::string VariableDeclare::build_opencl(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + type + " " + name + (this->val ? " = " + this->val->build_opencl(0) : "");
}

std::string VariableDeclare::build_metal(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + type + " " + name + (this->val ? " = " + this->val->build_metal(0) : "");
}

std::string VariableDeclare::build_cuda(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + type + " " + name + (this->val ? " = " + this->val->build_cuda(0) : "");
}

std::string VariableDeclare::build_cpu(const std::size_t & indentation) const {
    return ""; 
}