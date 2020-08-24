#pragma once

#if GPGPU_CUDA
#include <jitify.hpp>
#endif

#include "runtime.hpp"
#include "builder.hpp"

namespace gpgpu {
	class Kernel {
	public:
		Kernel() = default;
		~Kernel() = default;

		static bool has_cuda() {
#ifdef GPGPU_CUDA
			return true;
#else
			return false;
#endif
		}

		static bool has_opencl() {
#ifdef GPGPU_OPENCL
			return true;
#else
			return false;
#endif
		}

		static bool has_metal() {
#ifdef GPGPU_METAL
			return true;
#else
			return false;
#endif
		}

		static bool has_cpu() {
			return true;
		}

		static Runtime getPreferredRuntime()  {
			if (Kernel::has_metal()) {
				return Runtime::Metal;
			} else if (Kernel::has_cuda()) {
				return Runtime::CUDA;
			} else if (Kernel::has_opencl()) {
				return Runtime::OpenCL;
			} else {
				return Runtime::CPU;
			}
		}

		static Builder GetBuilder() {
			return Builder(Kernel::getPreferredRuntime());
		}

		static Builder GetBuilderFor(const Runtime& rt) {
			return Builder(rt);
		}
	};
}
