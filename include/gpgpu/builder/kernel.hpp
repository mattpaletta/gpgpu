#pragma once

#include "line_comment.hpp"
#include "global_thread.hpp"
#include "integer_constant.hpp"
#include "variable_declare.hpp"
#include "base_builder.hpp"
#include "function_arg.hpp"
#include "../runtime.hpp"

#include <string>
#include <vector>
#include <memory>

namespace gpgpu::builder {
	class Kernel : public BaseBuilder {
	public:
		Kernel(const std::string& _name, std::vector<std::unique_ptr<FunctionArg>>&& _args, const std::string& _returnType);
		~Kernel() = default;

		template<typename... Args>
		void addArg(const std::size_t& i, Args... args) {
			this->args.emplace(this->args.begin() + i, std::make_unique<FunctionArg>(args...));
		}

		template<typename... Args>
		void addArg(Args... args) {
			this->args.emplace_back(std::make_unique<FunctionArg>(args...));
		}

		template<typename... Args>
		void addComment(Args... args) {
			this->body.emplace_back(std::make_unique<LineComment>(args...));
		}

		template<typename... Args>
		void addGlobalThread(Args... args) {
			this->body.emplace_back(std::make_unique<GlobalThread>(args...));
		}

		std::unique_ptr<GlobalThread> getGlobalThread(int gthread) const;

		void addGlobalThread(int gthread);
		void addVariable(const std::string& type, const std::string& name, std::unique_ptr<BaseBuilder>&& val);
		void addBody(std::unique_ptr<BaseBuilder>&& element);

		std::string getName() const;

		virtual std::string build_opencl(const std::size_t& indentation = 0) const override;
		virtual std::string build_cuda(const std::size_t& indentation = 0) const override;
		virtual std::string build_metal(const std::size_t& indentation = 0) const override;
	private:
		std::string name;
		std::vector<std::unique_ptr<FunctionArg>> args;
		std::string returnType;

		std::vector<std::unique_ptr<BaseBuilder>> body;

		std::string getArgsStr(const Runtime& rt) const;
		std::string getBody(const std::size_t& indentation, const Runtime& rt) const;
	};
}
