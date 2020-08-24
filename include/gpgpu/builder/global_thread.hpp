//
//  global_thread.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once

#include "base_builder.hpp"
#include "kernel.hpp"
#include "constants.hpp"

namespace gpgpu::builder {
    class GlobalThread : public BaseBuilder {
        std::unique_ptr<BaseBuilder> body;
    public:
        GlobalThread(std::unique_ptr<BaseBuilder>&& _body) : BaseBuilder(), body(std::move(_body)) {}
        ~GlobalThread() = default;

        std::string build_opencl(const std::size_t& indentation) const override {
            return this->getIndentation(indentation) + "get_global_id(" + this->body->build_opencl(0) + ")";
        }

        std::string build_metal(const std::size_t& indentation) const override {
            return this->getIndentation(indentation) + constants::metal_id_var_name;
        }
    };
}
