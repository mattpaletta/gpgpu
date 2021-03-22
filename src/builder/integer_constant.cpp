#include <gpgpu/builder/integer_constant.hpp>
using namespace gpgpu::builder;

IntegerConstant::IntegerConstant(const int _val) : BaseBuilder(), val(_val) {}

std::string IntegerConstant::build_shared(const std::size_t& indentation) const {
    return this->getIndentation(indentation) + std::to_string(this->val);
}

std::string IntegerConstant::build_opencl(const std::size_t & indentation) const { 
    return this->build_shared(indentation); 
}

std::string IntegerConstant::build_metal(const std::size_t & indentation) const {
    return this->build_shared(indentation); 
}

std::string IntegerConstant::build_cuda(const std::size_t & indentation) const { 
    return this->build_shared(indentation); 
}
