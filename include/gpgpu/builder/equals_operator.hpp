//
//  equals_operator.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//
#pragma once
#include "base_builder.hpp"

#include <memory>

namespace gpgpu::builder {
	class EqualsOperator : public BaseBuilder {
	public:
		EqualsOperator(std::unique_ptr<BaseBuilder>&& _lhs, std::unique_ptr<BaseBuilder>&& _rhs);
		~EqualsOperator() = default;

		std::string build_opencl(const std::size_t& indentation) const override;
		std::string build_metal(const std::size_t& indentation) const override;
		std::string build_cuda(const std::size_t& indentation) const override;

	private:
		std::unique_ptr<BaseBuilder> lhs;
		std::unique_ptr<BaseBuilder> rhs;
	};
}
