#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <iostream>

#include <gpgpu/kernel.hpp>
#include <gpgpu/builder/array_element.hpp>
#include <gpgpu/builder/variable_reference.hpp>
#include <gpgpu/builder/equals_operator.hpp>
#include <gpgpu/builder/addition_operator.hpp>

TEST_CASE("test_runtime", "[kernel]") {
	CHECK((gpgpu::Kernel::has_cuda() || gpgpu::Kernel::has_opencl() || gpgpu::Kernel::has_metal() || gpgpu::Kernel::has_cpu()));
}
template<class T>
std::vector<T> range(const std::size_t& start, const std::size_t& end, const std::size_t& step) {
    std::vector<T> out;
    out.reserve((end - start) / step);
    for (std::size_t i = start; i < end; i += step) {
        out.push_back(i);
    }
    return out;
}

template<class T>
std::vector<T> range(const std::size_t& start, const std::size_t& end) { return range<T>(start, end, 1); }
template<class T>
std::vector<T> range(const std::size_t& end) { return range<T>(0, end); }

TEST_CASE("build kernel", "[builder]") {
	auto builder = gpgpu::Kernel::GetBuilder();
	auto* kernel = builder.NewKernel("vector_add", {}, "void");
	kernel->addArg(std::make_unique<gpgpu::builder::FunctionArg>(gpgpu::builder::ARG_TYPE::GLOBAL, "int*", "A", true));
	kernel->addArg(std::make_unique<gpgpu::builder::FunctionArg>(gpgpu::builder::ARG_TYPE::GLOBAL, "int*", "B", true));
	kernel->addArg(std::make_unique<gpgpu::builder::FunctionArg>(gpgpu::builder::ARG_TYPE::GLOBAL, "int*", "C", false));

	kernel->addComment({"These comments must be closed because they will be appended to one line"});

    kernel->addComment({"Get the index of the current element to be processed"});
    kernel->addVariable("int", "i", std::make_unique<gpgpu::builder::GlobalThread>(std::make_unique<gpgpu::builder::IntegerConstant>(0)));

    auto a_i = std::make_unique<gpgpu::builder::ArrayElement>(std::make_unique<gpgpu::builder::VariableReference>("A"), std::make_unique<gpgpu::builder::VariableReference>("i")); // A[i];
    auto b_i = std::make_unique<gpgpu::builder::ArrayElement>(std::make_unique<gpgpu::builder::VariableReference>("B"), std::make_unique<gpgpu::builder::VariableReference>("i")); // B[i];
    auto c_i = std::make_unique<gpgpu::builder::ArrayElement>(std::make_unique<gpgpu::builder::VariableReference>("C"), std::make_unique<gpgpu::builder::VariableReference>("i")); // C[i];

    auto a_plus_b = std::make_unique<gpgpu::builder::AdditionOperator>(std::move(a_i), std::move(b_i));
    auto equals_c = std::make_unique<gpgpu::builder::EqualsOperator>(std::move(c_i), std::move(a_plus_b));

    kernel->addComment({"Do the operation"});
    kernel->addBody(std::move(equals_c));

    std::cout << builder.dump(gpgpu::CUDA) << std::endl;

    const auto A = range<int>(100);
    const auto B = range<int>(100, 200);
    std::vector<int> C;
    CHECK_NOTHROW(builder.run(gpgpu::CUDA, "vector_add", &C, A, B));
    for (const auto& i : C) {
        std::cout << i << " ";
    }
}
