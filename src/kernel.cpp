#include <gpgpu/kernel.hpp>
#include <stdexcept>

using namespace gpgpu;

Kernel::Kernel() = default;

bool Kernel::has_cuda() {
#ifdef GPGPU_CUDA
	return true;
#else
	return false;
#endif
}

bool Kernel::has_opencl() {
#ifdef GPGPU_OPENCL
	return true;
#else
	return false;
#endif
}

bool Kernel::has_metal() {
#ifdef GPGPU_METAL
	return true;
#else
	return false;
#endif
}

Runtime Kernel::getPreferredRuntime() {
	if (Kernel::has_metal()) {
		return Runtime::Metal;
	} else if (Kernel::has_cuda()) {
		return Runtime::CUDA;
	} else if (Kernel::has_opencl()) {
		return Runtime::OpenCL;
	} else {
		throw std::runtime_error("Unknown runtime");
	}
}

Builder Kernel::GetBuilder() {
	return Builder(Kernel::getPreferredRuntime());
}

Builder Kernel::GetBuilderFor(const Runtime& rt) {
	return Builder(rt);
}
