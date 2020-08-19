#pragma once
#include <string>
#include <vector>

#include "line_comment.hpp"

namespace gpgpu::builder {
	class Kernel {
		using argType = std::string;

		std::string name;
		std::vector<argType> args;
		std::string returnType;

		std::vector<std::string> body;
	
		std::string getArgsStr() const {
			std::string argsLine;
			for (std::size_t i = 0; i < this->args.size() - 1; ++i) {
				argsLine += this->args.at(i) + ", ";
			}
			if (!this->args.empty()) {
				// No trailing comma
				argsLine += this->args.at(this->args.size() - 1);
			}
			return "(" + argsLine + ")";
		}

		std::string getBody() const {
			// TODO: Write a generic vector_to_string method
			std::string bodyLine;
			for (std::size_t i = 0; i < this->body.size(); ++i) {
				bodyLine += this->body.at(i) + "\n";
			}
			return bodyLine;
		}

	public:

		Kernel(const std::string& _name, const std::vector<argType>& _args, const std::string& _returnType) : name(_name), args(_args), returnType(_returnType) {}
		~Kernel() = default;

		void addArg(const argType& arg, const std::size_t& i = 0) {
			this->args.insert(this->args.begin() + i, arg);
		}

		void addComment(const LineComment& comment) {
			// TODO: make this lazy.
			this->body.emplace_back(comment.build_opencl());
		}

		std::string getName() const {
			return this->name;
		}

		std::string build_opencl() const {
			const std::string header = "__kernel " + this->returnType + " " + this->getName() + this->getArgsStr() + " {" + "\n";
			const std::string end = "}";
			return header + this->getBody() + end;
		}
	};
}