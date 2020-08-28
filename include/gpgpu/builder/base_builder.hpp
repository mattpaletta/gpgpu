//
//  base_builder.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once
#include <string>

namespace gpgpu {
    namespace builder {
        class BaseBuilder {
        protected:
            std::string getIndentation(const std::size_t& indentation) const {
                std::string out;
                for (int i = 0; i < indentation; ++i) {
                    out += "    ";
                }
                return out;
            }
        public:
            BaseBuilder() = default;
            virtual ~BaseBuilder() = default;

            virtual bool isComment() const { return false; }

            virtual std::string build_opencl(const std::size_t& indentation) const { return ""; }
            virtual std::string build_metal(const std::size_t& indentation) const { return ""; }
            virtual std::string build_cuda(const std::size_t& indentation) const { return ""; }
            virtual std::string build_cpu(const std::size_t& indentation) const { return ""; }
        };
    }
}
