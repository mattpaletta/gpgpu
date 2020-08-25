//
//  function_arg.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-21.
//

#pragma once

#include "base_builder.hpp"

namespace gpgpu::builder {
    enum ARG_TYPE { GLOBAL = 0, DEVICE, HOST };

    class FunctionArg : public BaseBuilder {
        ARG_TYPE location;

        std::string type;
        std::string name;

        bool is_const;

        std::unique_ptr<BaseBuilder> val;

        std::string opencl_location_to_string() const {
            switch (this->location) {
                case GLOBAL:
                    return "global";
                case DEVICE:
                    return "device";
                case HOST:
                    return "host";
            }
        }

        std::string metal_location_to_string() const {
            switch (this->location) {
                case GLOBAL:
                    // Metal copies to the device
                    return "device";
                case DEVICE:
                    return "device";
                case HOST:
                    return "host";
            }
        }

        std::string cuda_location_to_string() const {
            switch (this->location) {
                case GLOBAL:
                    return "__global__";
                case DEVICE:
                    return "__device__";
                case HOST:
                    return "__host__";
            }
        }

    public:
        FunctionArg(const ARG_TYPE& _location, const std::string& _type, const std::string& _name, const bool _is_const) : BaseBuilder(), location(_location), type(_type), is_const(_is_const), name(_name), val() {}
        ~FunctionArg() = default;

        std::string build_opencl(const std::size_t& indentation, const std::size_t& i) const {
            return this->getIndentation(0) + this->opencl_location_to_string() + " " + (this->is_const ? "const " : "") + this->type + " " + name + (this->val ? " = " + this->val->build_opencl(0) : "");
        }

        std::string build_metal(const std::size_t& indentation, const std::size_t& i) const {
            return this->getIndentation(0) + (this->is_const ? "const " : "") + this->metal_location_to_string() + " " + this->type + " " + name + + "[[ buffer(" + std::to_string(i) + ") ]]" + (this->val ? " = " + this->val->build_metal(0) : "");
        }

        std::string build_cuda(const std::size_t& indentation, const std::size_t& i) const {
            return this->getIndentation(0) + this->cuda_location_to_string() + " " + this->type + " " + name + (this->val ? " = " + this->val->build_cuda(0) : "");
        }

        std::string build_cpu(const std::size_t& indentation, const std::size_t& i) const { return ""; }
    };
}
