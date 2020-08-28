//
//  IntegerConstant.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once
#include "base_builder.hpp"

namespace gpgpu {
    namespace builder {
        class IntegerConstant : public BaseBuilder {
        private:
            int val;

            std::string build_shared(const std::size_t& indentation) const;
        public:
            IntegerConstant(const int _val);
            ~IntegerConstant() = default;

            std::string build_opencl(const std::size_t& indentation) const override;
            std::string build_metal(const std::size_t& indentation) const override;
            std::string build_cuda(const std::size_t& indentation) const override;
            std::string build_cpu(const std::size_t& indentation) const override;
        };
    }
}