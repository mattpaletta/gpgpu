#pragma once
#include <string>
#include <vector>

#include "builder/line_comment.hpp"
#include "builder/kernel.hpp"
#include "runtime.hpp"

namespace gpgpu {
	class Builder {
	private:
		std::vector<builder::Kernel> funcs;
		const Runtime rt;

	public:
		Builder(const Runtime& _rt) : rt(_rt) {};
		~Builder() = default;

		builder::Kernel* GetKernel(const std::string& name) {
			for (auto& f : this->funcs) {
				if (f.getName() == name) {
					return &f;
				}
			}
			return nullptr;
		}

		builder::Kernel* NewKernel(const std::string& name, const std::vector<std::string>& args, const std::string& returnType) {
			this->funcs.emplace_back(name, args, returnType);
			return &this->funcs.back();
		}
	};
}