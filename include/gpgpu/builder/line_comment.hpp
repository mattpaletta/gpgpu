#pragma once
#include <string>
#include <vector>

namespace gpgpu::builder {
	class LineComment {
	private:
		std::string text;

		std::string build_shared() const {
			return "//" + this->text;
		}

	public:
		LineComment(const std::string& _text) : text(_text) {}
		~LineComment() = default;

		std::string build_opencl() const { return this->build_shared(); }
		std::string build_metal() const {return this->build_shared(); }
		std::string build_cuda() const { return this->build_shared(); }
		std::string build_cpu() const { return ""; }
	};
}