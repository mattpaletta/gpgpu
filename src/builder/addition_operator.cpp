#include <gpgpu/builder/addition_operator.hpp>

using namespace gpgpu::builder;

AdditionOperator::AdditionOperator(std::unique_ptr<BaseBuilder>&& _lhs, std::unique_ptr<BaseBuilder>&& _rhs) : BaseBuilder(), lhs(std::move(_lhs)), rhs(std::move(_rhs)) {}

std::string AdditionOperator::build_opencl(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + this->lhs->build_opencl(0) + " + " + this->rhs->build_opencl(0);
}

std::string AdditionOperator::build_metal(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + this->lhs->build_metal(0) + " + " + this->rhs->build_metal(0);
}

std::string AdditionOperator::build_cuda(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + this->lhs->build_cuda(0) + " + " + this->rhs->build_cuda(0);
}
