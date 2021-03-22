//
//  array_element.hpp
//  gpgpu
//
//  Created by Matthew Paletta on 2020-08-19.
//

#pragma once

#include "base_builder.hpp"

#include <memory>

namespace gpgpu::builder {
	class ArrayElement : public BaseBuilder {
	public:
		ArrayElement(std::unique_ptr<BaseBuilder>&& _lst, std::unique_ptr<BaseBuilder>&& _element);
		~ArrayElement() = default;

		std::string build_opencl(const std::size_t& indentation) const override;
		std::string build_metal(const std::size_t& indentation) const override;
		std::string build_cuda(const std::size_t& indentation) const override;
	private:
		std::unique_ptr<BaseBuilder> lst; // Likely to be a variable
		std::unique_ptr<BaseBuilder> element; // Inside the square brackets, could be an expression

	};
}
