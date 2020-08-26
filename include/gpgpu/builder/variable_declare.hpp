//
//  variable_declare.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once

#include "base_builder.hpp"

namespace gpgpu {
    namespace builder {
        class VariableDeclare : public BaseBuilder {
            std::string type;
            std::string name;

            std::unique_ptr<BaseBuilder> val;

        public:
            VariableDeclare(const std::string& _type, const std::string& _name, std::unique_ptr<BaseBuilder>&& _val) : BaseBuilder(), type(_type), name(_name), val(std::move(_val)) {}
            VariableDeclare(const std::string& _type, const std::string& _name) : BaseBuilder(), type(_type), name(_name), val() {}
            ~VariableDeclare() = default;

            std::string build_opencl(const std::size_t& indentation) const override {
                return this->getIndentation(indentation) + type + " " + name + (this->val ? " = " + this->val->build_opencl(0) : "");
            }

            std::string build_metal(const std::size_t& indentation) const override {
                return this->getIndentation(indentation) + type + " " + name + (this->val ? " = " + this->val->build_metal(0) : "");
            }

            std::string build_cuda(const std::size_t& indentation) const override {
                return this->getIndentation(indentation) + type + " " + name + (this->val ? " = " + this->val->build_cuda(0) : "");
            }

            std::string build_cpu(const std::size_t& indentation) const override { return ""; }
        };
    }
}
