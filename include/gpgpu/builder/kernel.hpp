#pragma once
#include <string>
#include <vector>
#include <memory>

#include "line_comment.hpp"
#include "global_thread.hpp"
#include "integer_constant.hpp"
#include "variable_declare.hpp"
#include "base_builder.hpp"
#include "function_arg.hpp"
#include "../runtime.hpp"

namespace gpgpu {
    namespace builder {
        class Kernel : public BaseBuilder {
        public:
        private:
            std::string name;
            std::vector<std::unique_ptr<FunctionArg>> args;
            std::string returnType;

            std::vector<std::unique_ptr<BaseBuilder>> body;

            std::string getArgsStr(const Runtime& rt) const;
            std::string getBody(const std::size_t& indentation, const Runtime& rt) const;
        public:

            Kernel(const std::string& _name, std::vector<std::unique_ptr<FunctionArg>>&& _args, const std::string& _returnType);
            ~Kernel() = default;

            void addArg(std::unique_ptr<FunctionArg>&& arg, const std::size_t& i) {
                this->args.emplace(this->args.begin() + i, std::move(arg));
            }

            void addArg(std::unique_ptr<FunctionArg>&& arg) {
                this->args.emplace_back(std::move(arg));
            }

            void addComment(LineComment&& comment) {
                this->body.emplace_back(std::make_unique<LineComment>(comment));
            }

            void addGlobalThread(std::unique_ptr<GlobalThread>&& gthread) {
                this->body.emplace_back(std::move(gthread));
            }

            void addGlobalThread(int gthread) {
                this->addGlobalThread(std::move(std::make_unique<GlobalThread>(std::make_unique<IntegerConstant>(gthread))));
            }

            void addVariable(const std::string& type, const std::string& name, std::unique_ptr<BaseBuilder>&& val) {
                this->body.emplace_back(std::make_unique<VariableDeclare>(type, name, std::move(val)));
            }

            void addBody(std::unique_ptr<BaseBuilder>&& element) {
                this->body.emplace_back(std::move(element));
            }

            std::string getName() const;

            virtual std::string build_opencl(const std::size_t& indentation = 0) const override;
            virtual std::string build_cuda(const std::size_t& indentation = 0) const override;
            virtual std::string build_metal(const std::size_t& indentation = 0) const override;
        };
    }
}