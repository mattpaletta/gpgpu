//
//  function_arg.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-21.
//

#pragma once
#include <memory>
#include "base_builder.hpp"

namespace gpgpu {
    namespace builder {
        enum ARG_TYPE { GLOBAL = 0, DEVICE, HOST };

        class FunctionArg : public BaseBuilder {
            ARG_TYPE location;

            std::string type;
            std::string name;

            bool is_const;

            std::unique_ptr<BaseBuilder> val;

            std::string opencl_location_to_string() const;
            std::string metal_location_to_string() const;
            std::string cuda_location_to_string() const;

            std::string constStr() const;
        public:
            FunctionArg(const ARG_TYPE& _location, const std::string& _type, const std::string& _name, const bool _is_const);
            ~FunctionArg() = default;

            std::string build_opencl_arg(const std::size_t& indentation, const std::size_t& i) const;
            std::string build_metal_arg(const std::size_t& indentation, const std::size_t& i) const;
            std::string build_cuda_arg(const std::size_t& indentation, const std::size_t& i) const;
            std::string build_cpu_arg(const std::size_t& indentation, const std::size_t& i) const;
        };
    }
}