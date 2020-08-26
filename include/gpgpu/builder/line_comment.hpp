#pragma once
#include <string>
#include <vector>

#include "base_builder.hpp"

namespace gpgpu {
	namespace builder {
		class LineComment : public BaseBuilder {
		private:
			std::string text;

			std::string build_shared(const std::size_t& indentation) const {
				return this->getIndentation(indentation) + "// " + this->text;
			}
		public:
			LineComment(const std::string& _text) : BaseBuilder(), text(_text) {}
			~LineComment() = default;

			virtual bool isComment() const override { return true; }

			std::string build_opencl(const std::size_t& indentation) const override { return this->build_shared(indentation); }
			std::string build_metal(const std::size_t& indentation) const override { return this->build_shared(indentation); }
			std::string build_cuda(const std::size_t& indentation) const override { return this->build_shared(indentation); }
			std::string build_cpu(const std::size_t& indentation) const override { return ""; }
		};
	}
}
