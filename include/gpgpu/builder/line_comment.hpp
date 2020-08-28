#pragma once
#include <string>
#include <vector>

#include "base_builder.hpp"

namespace gpgpu {
	namespace builder {
		class LineComment : public BaseBuilder {
		private:
			std::string text;

			std::string build_shared(const std::size_t& indentation) const;
		public:
			LineComment(const std::string& _text);
			~LineComment() = default;

			virtual bool isComment() const override;

			std::string build_opencl(const std::size_t& indentation) const override;
			std::string build_metal(const std::size_t& indentation) const override;
			std::string build_cuda(const std::size_t& indentation) const override;
			std::string build_cpu(const std::size_t& indentation) const override;
		};
	}
}
