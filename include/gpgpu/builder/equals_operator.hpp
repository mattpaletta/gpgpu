//
//  equals_operator.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//
#pragma once

#include "base_builder.hpp"

namespace gpgpu::builder {
    class EqualsOperator : public BaseBuilder {
        std::unique_ptr<BaseBuilder> lhs;
        std::unique_ptr<BaseBuilder> rhs;

        std::string build_shared(const std::size_t& indentation) const {
            return this->getIndentation(indentation) + this->lhs->build_opencl(0) + " = " + this->rhs->build_opencl(0);
        }

    public:
        EqualsOperator(std::unique_ptr<BaseBuilder>&& _lhs, std::unique_ptr<BaseBuilder>&& _rhs) : BaseBuilder(), lhs(std::move(_lhs)), rhs(std::move(_rhs)) {}
        ~EqualsOperator() = default;

        std::string build_opencl(const std::size_t& indentation) const override { return this->build_shared(indentation); }
        std::string build_metal(const std::size_t& indentation) const override { return this->build_shared(indentation); }
        std::string build_cuda(const std::size_t& indentation) const override { return this->build_shared(indentation); }
        std::string build_cpu(const std::size_t& indentation) const override { return ""; }
    };
}
