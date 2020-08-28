//
//  variable_declare.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once
#include <memory>
#include "base_builder.hpp"

namespace gpgpu {
    namespace builder {
        class VariableDeclare : public BaseBuilder {
        private:
            std::string type;
            std::string name;

            std::unique_ptr<BaseBuilder> val;
        public:
            VariableDeclare(const std::string& _type, const std::string& _name, std::unique_ptr<BaseBuilder>&& _val);
            VariableDeclare(const std::string& _type, const std::string& _name);
            ~VariableDeclare() = default;

            std::string build_opencl(const std::size_t& indentation) const override;
            std::string build_metal(const std::size_t& indentation) const override;
            std::string build_cuda(const std::size_t& indentation) const override;
            std::string build_cpu(const std::size_t& indentation) const override;
        };
    }
}
