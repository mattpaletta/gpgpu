//
//  addition_operator.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once
#include <memory>
#include "base_builder.hpp"

namespace gpgpu {
    namespace builder {
        class AdditionOperator : public BaseBuilder {
            std::unique_ptr<BaseBuilder> lhs;
            std::unique_ptr<BaseBuilder> rhs;

        public:
            AdditionOperator(std::unique_ptr<BaseBuilder>&& _lhs, std::unique_ptr<BaseBuilder>&& _rhs);
            ~AdditionOperator() = default;

            std::string build_opencl(const std::size_t& indentation) const override;
            std::string build_metal(const std::size_t& indentation) const override;
            std::string build_cuda(const std::size_t& indentation) const override;
            std::string build_cpu(const std::size_t& indentation) const override;
        };
    }
}