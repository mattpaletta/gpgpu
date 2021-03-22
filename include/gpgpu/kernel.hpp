#pragma once

#if GPGPU_CUDA
#include <jitify.hpp>
#endif

#include "runtime.hpp"
#include "builder.hpp"

namespace gpgpu {
	class Kernel {
	public:
		Kernel();
		~Kernel() = default;

		static bool has_cuda();
		static bool has_opencl();
		static bool has_metal();

		static Runtime getPreferredRuntime();
		static Builder GetBuilder();
		static Builder GetBuilderFor(const Runtime& rt);
	};
}
