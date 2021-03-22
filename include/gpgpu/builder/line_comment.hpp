#pragma once

#include "base_builder.hpp"

#include <string>

namespace gpgpu::builder {
	class LineComment : public BaseBuilder {
	public:
		LineComment(const std::string& _text);
		~LineComment() = default;

		virtual bool isComment() const override;

		std::string build_opencl(const std::size_t& indentation) const override;
		std::string build_metal(const std::size_t& indentation) const override;
		std::string build_cuda(const std::size_t& indentation) const override;
	private:
		std::string text;

		std::string build_shared(const std::size_t& indentation) const;
	};
}
