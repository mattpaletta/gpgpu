//
//  array_element.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once

#include "base_builder.hpp"

namespace gpgpu::builder {
    class ArrayElement : public BaseBuilder {
        std::unique_ptr<BaseBuilder> lst; // Likely to be a variable
        std::unique_ptr<BaseBuilder> element; // Inside the square brackets, could be an expression

    public:
        ArrayElement(std::unique_ptr<BaseBuilder>&& _lst, std::unique_ptr<BaseBuilder>&& _element) : BaseBuilder(), lst(std::move(_lst)), element(std::move(_element)) {}
        ~ArrayElement() = default;

        std::string build_opencl(const std::size_t& indentation) const override { 
            return this->getIndentation(indentation) + this->lst->build_opencl(0) + "[" + this->element->build_opencl(0) + "]";
        }

        std::string build_metal(const std::size_t& indentation) const override { 
            return this->getIndentation(indentation) + this->lst->build_metal(0) + "[" + this->element->build_metal(0) + "]";
        }

        std::string build_cuda(const std::size_t& indentation) const override { 
            return this->getIndentation(indentation) + this->lst->build_cuda(0) + "[" + this->element->build_cuda(0) + "]";
        }

        std::string build_cpu(const std::size_t& indentation) const override { return ""; }
    };
}
