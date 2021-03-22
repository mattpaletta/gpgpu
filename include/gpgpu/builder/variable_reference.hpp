//
//  variable_reference.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once
#include "base_builder.hpp"

#include <string>

namespace gpgpu::builder {
	class VariableReference : public BaseBuilder {

	public:
		VariableReference(const std::string& _var);
		~VariableReference() = default;

		std::string build_opencl(const std::size_t& indentation) const override;
		std::string build_metal(const std::size_t& indentation) const override;
		std::string build_cuda(const std::size_t& indentation) const override;
	private:
		std::string var;

		std::string build_shared(const std::size_t& indentation) const;
	};
}
