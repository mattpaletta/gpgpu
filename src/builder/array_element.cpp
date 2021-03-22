#include <gpgpu/builder/array_element.hpp>
using namespace gpgpu::builder;

ArrayElement::ArrayElement(std::unique_ptr<BaseBuilder>&& _lst, std::unique_ptr<BaseBuilder>&& _element) : BaseBuilder(), lst(std::move(_lst)), element(std::move(_element)) {}

std::string ArrayElement::build_opencl(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + this->lst->build_opencl(0) + "[" + this->element->build_opencl(0) + "]";
}

std::string ArrayElement::build_metal(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + this->lst->build_metal(0) + "[" + this->element->build_metal(0) + "]";
}

std::string ArrayElement::build_cuda(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + this->lst->build_cuda(0) + "[" + this->element->build_cuda(0) + "]";
}
