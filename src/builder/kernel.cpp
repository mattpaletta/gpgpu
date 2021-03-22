#include <gpgpu/builder/kernel.hpp>
#include <gpgpu/builder/constants.hpp>
using namespace gpgpu::builder;


Kernel::Kernel(const std::string& _name, std::vector<std::unique_ptr<FunctionArg>>&& _args, const std::string& _returnType) : BaseBuilder(), name(_name), args(std::move(_args)), returnType(_returnType) {}

std::string Kernel::getArgsStr(const Runtime& rt) const {
    std::string argsLine;
    for (std::size_t i = 0; i < this->args.size() - 1; ++i) {
        switch (rt) {
        case CUDA:
            argsLine += this->args.at(i)->build_cuda_arg(0, i) + ", ";
            break;
        case Metal:
            argsLine += this->args.at(i)->build_metal_arg(0, i) + ", ";
            break;
        case OpenCL:
            argsLine += this->args.at(i)->build_opencl_arg(0, i) + ", ";
            break;
        }
    }

    if (!this->args.empty()) {
        const auto i = this->args.size() - 1;
        // No trailing comma
        switch (rt) {
        case CUDA:
            argsLine += this->args.at(i)->build_cuda_arg(0, i);
            break;
        case Metal:
            argsLine += this->args.at(i)->build_metal_arg(0, i);
            break;
        case OpenCL:
            argsLine += this->args.at(i)->build_opencl_arg(0, i);
            break;
        }
    }

    if (rt == Metal) {
        // Add hidden parameter
        argsLine += ", uint " + constants::metal_id_var_name + " [[ thread_position_in_grid ]]";
    }

    return "(" + argsLine + ")";
}

void Kernel::addGlobalThread(int gthread) {
	this->body.emplace_back(this->getGlobalThread(gthread));
}

void Kernel::addVariable(const std::string& type, const std::string& name, std::unique_ptr<BaseBuilder>&& val) {
	this->body.emplace_back(std::make_unique<VariableDeclare>(type, name, std::move(val)));
}

void Kernel::addBody(std::unique_ptr<BaseBuilder>&& element) {
	this->body.emplace_back(std::move(element));
}

std::string Kernel::getBody(const std::size_t& indentation, const Runtime& rt) const {
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
        }
    }
    return bodyLine;
}

std::string Kernel::getName() const {
    return this->name;
}

auto Kernel::getGlobalThread(int gthread) const -> std::unique_ptr<GlobalThread> {
	return std::make_unique<GlobalThread>(std::make_unique<IntegerConstant>(gthread));
}

std::string Kernel::build_opencl(const std::size_t& indentation) const {
    const std::string header = this->getIndentation(indentation) + this->returnType + " kernel " + this->getName() + this->getArgsStr(OpenCL) + " {" + "\n";
    const std::string end = "}";
    return header + this->getBody(indentation + 1, OpenCL) + end;
}

std::string Kernel::build_cuda(const std::size_t& indentation) const {
    const std::string header = this->getIndentation(indentation) + "__global__ " + this->returnType + " " + this->getName() + this->getArgsStr(CUDA) + " {" + "\n";
    const std::string end = "}";
    return header + this->getBody(indentation + 1, CUDA) + end;
}

std::string Kernel::build_metal(const std::size_t& indentation) const {
    const std::string header = this->getIndentation(indentation) + "kernel " + this->returnType + " " + this->getName() + this->getArgsStr(Metal) + " {" + "\n";
    const std::string end = "}";
    return header + this->getBody(indentation + 1, Metal) + end;
}
