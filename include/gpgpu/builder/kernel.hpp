#pragma once
#include <string>
#include <vector>
#include <memory>

#include "gpgpu/builder/line_comment.hpp"
#include "gpgpu/builder/global_thread.hpp"
#include "gpgpu/builder/IntegerConstant.hpp"
#include "gpgpu/builder/variable_declare.hpp"
#include "gpgpu/builder/base_builder.hpp"
#include "gpgpu/builder/function_arg.hpp"
#include "gpgpu/builder.hpp"
#include "gpgpu/runtime.hpp"
#include "constants.hpp"

namespace gpgpu {
    namespace builder {
        class Kernel : public BaseBuilder {
        public:
        private:
            std::string name;
            std::vector<std::unique_ptr<FunctionArg>> args;
            std::string returnType;

            std::vector<std::unique_ptr<BaseBuilder>> body;

            std::string getArgsStr(const Runtime& rt) const {
                std::string argsLine;
                for (std::size_t i = 0; i < this->args.size() - 1; ++i) {
                    switch (rt) {
                    case CUDA:
                        argsLine += this->args.at(i)->build_cuda(0, i) + ", ";
                        break;
                    case Metal:
                        argsLine += this->args.at(i)->build_metal(0, i) + ", ";
                        break;
                    case OpenCL:
                        argsLine += this->args.at(i)->build_opencl(0, i) + ", ";
                        break;
                    case CPU:
                        argsLine += this->args.at(i)->build_cpu(0, i) + ", ";
                        break;
                    }
                }

                if (!this->args.empty()) {
                    const auto i = this->args.size() - 1;
                    // No trailing comma
                    switch (rt) {
                    case CUDA:
                        argsLine += this->args.at(i)->build_cuda(0, i);
                        break;
                    case Metal:
                        argsLine += this->args.at(i)->build_metal(0, i);
                        break;
                    case OpenCL:
                        argsLine += this->args.at(i)->build_opencl(0, i);
                        break;
                    case CPU:
                        argsLine += this->args.at(i)->build_cpu(0, i);
                        break;
                    }
                }

                if (rt == Metal) {
                    // Add hidden parameter
                    argsLine += ", uint " + constants::metal_id_var_name + " [[ thread_position_in_grid ]]";
                }

                return "(" + argsLine + ")";
            }

            std::string getBody(const std::size_t& indentation, const Runtime& rt) const {
                // TODO: Write a generic vector_to_string method
                std::string bodyLine;
                for (std::size_t i = 0; i < this->body.size(); ++i) {
                    const std::string endSemicolon = std::string{ this->body.at(i)->isComment() ? "" : ";" } + "\n";
                    switch (rt) {
                    case CUDA:
                        bodyLine += this->body.at(i)->build_cuda(indentation) + endSemicolon;
                        break;
                    case Metal:
                        bodyLine += this->body.at(i)->build_metal(indentation) + endSemicolon;
                        break;
                    case OpenCL:
                        bodyLine += this->body.at(i)->build_opencl(indentation) + endSemicolon;
                        break;
                    case CPU:
                        bodyLine += this->body.at(i)->build_cpu(indentation) + endSemicolon;
                        break;
                    }
                }
                return bodyLine;
            }

        public:

            Kernel(const std::string& _name, std::vector<std::unique_ptr<FunctionArg>>&& _args, const std::string& _returnType) : BaseBuilder(), name(_name), args(std::move(_args)), returnType(_returnType) {}
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

            std::string getName() const {
                return this->name;
            }

            virtual std::string build_opencl(const std::size_t& indentation = 0) const override {
                const std::string header = this->getIndentation(indentation) + this->returnType + " kernel " + this->getName() + this->getArgsStr(OpenCL) + " {" + "\n";
                const std::string end = "}";
                return header + this->getBody(indentation + 1, OpenCL) + end;
            }

            virtual std::string build_cuda(const std::size_t& indentation = 0) const override {
                const std::string header = this->getIndentation(indentation) + "__global__ " + this->returnType + " " + this->getName() + this->getArgsStr(CUDA) + " {" + "\n";
                const std::string end = "}";
                return header + this->getBody(indentation + 1, CUDA) + end;
            }

            virtual std::string build_metal(const std::size_t& indentation = 0) const override {
                const std::string header = this->getIndentation(indentation) + "kernel " + this->returnType + " " + this->getName() + this->getArgsStr(Metal) + " {" + "\n";
                const std::string end = "}";
                return header + this->getBody(indentation + 1, Metal) + end;
            }
        };
    }
}