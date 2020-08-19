#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <iostream>

#include <gpgpu/kernel.hpp>

TEST_CASE("test_runtime", "[kernel]") {
	CHECK((gpgpu::Kernel::has_cuda() || gpgpu::Kernel::has_opencl() || gpgpu::Kernel::has_metal() || gpgpu::Kernel::has_cpu()));
}

TEST_CASE("build kernel", "[builder]") {
	auto builder = gpgpu::Kernel::GetBuilder();
	auto* kernel = builder.NewKernel("vector_add", {}, "void");
	kernel->addArg("__global const int* A");
	kernel->addArg("__global const int* B");
	kernel->addArg("__global int* C");

	kernel->addComment({"These comments must be closed because they will be appended to one line"});
	std::cout << kernel->build_opencl() << std::endl;
}