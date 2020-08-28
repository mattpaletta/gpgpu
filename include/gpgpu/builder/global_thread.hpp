//
//  global_thread.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once
#include <memory>
#include "base_builder.hpp"

namespace gpgpu {
    namespace builder {
        class GlobalThread : public BaseBuilder {
            std::unique_ptr<BaseBuilder> body;
        public:
            GlobalThread(std::unique_ptr<BaseBuilder>&& _body);
            ~GlobalThread() = default;

            std::string build_opencl(const std::size_t& indentation) const override;
            std::string build_cuda(const std::size_t& indentation) const override;
            std::string build_metal(const std::size_t& indentation) const override;
        };
    }
}