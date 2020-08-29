# GPGPU
This project is an expermient a creating a thin wrapper-library around the most popular GPU libraries (CUDA, OpenCL, and Metal).  The goal is to create an abstract API that other library/application developers can use when using GPU code.  This should enable them to write their algorithm once, and automatically run it on the hardware available in the users machine.  If there are multiple options, the developer can choose which mode they'd like, or the library will pick the preferred option.

## Motivation
This project started as both experimentation to see if I could do it, and because I felt that there were many amazing projects being written in CUDA, but I have a Mac.  Apple recently stopped supporting CUDA and OpenCL on MacOS, so I wanted a standard platform for me to write GPU code in my own projects that will enable me to accelerate them using the GPU in my Mac.  Theoretically, this would also work on iOS/Android, though this hasn't been tested.

## Build Status
[![Build Status](https://travis-ci.com/mattpaletta/gpgpu.svg?branch=master)](https://travis-ci.com/mattpaletta/gpgpu)
This is still a very experimental library.  I make no guarantees about a stable API or implementation.  Use at your own risk.

## Getting Started
Here's an example function that uses GPGPU to create a kernel that adds two lists and stores the result in a third list.  Note, this API is being cleaned up, and may appear different.  See `test/` for the most recent verison.
```cpp
#include <vector>
#include <gpgpu/kernel.hpp>
#include <gpgpu/builder/array_element.hpp>
#include <gpgpu/builder/variable_reference.hpp>
#include <gpgpu/builder/equals_operator.hpp>
#include <gpgpu/builder/addition_operator.hpp>

int main() {
	gpgpu::Builder builder = gpgpu::Kernel::GetBuilderFor(gpgpu::Runtime::CUDA);
	auto* kernel = builder.NewKernel("vector_add", {}, "void");

	kernel->addArg(gpgpu::builder::ARG_TYPE::GLOBAL, "int*", "A", true);
	kernel->addArg(gpgpu::builder::ARG_TYPE::GLOBAL, "int*", "B", true);
	kernel->addArg(gpgpu::builder::ARG_TYPE::GLOBAL, "int*", "C", false);

    	kernel->addComment("Get the index of the current element to be processed");
	kernel->addVariable("int", "i", std::move(kernel->getGlobalThread(0)));

	auto a_i = std::make_unique<gpgpu::builder::ArrayElement>(std::make_unique<gpgpu::builder::VariableReference>("A"), std::make_unique<gpgpu::builder::VariableReference>("i")); // A[i];
	auto b_i = std::make_unique<gpgpu::builder::ArrayElement>(std::make_unique<gpgpu::builder::VariableReference>("B"), std::make_unique<gpgpu::builder::VariableReference>("i")); // B[i];
	auto c_i = std::make_unique<gpgpu::builder::ArrayElement>(std::make_unique<gpgpu::builder::VariableReference>("C"), std::make_unique<gpgpu::builder::VariableReference>("i")); // C[i];

	auto a_plus_b = std::make_unique<gpgpu::builder::AdditionOperator>(std::move(a_i), std::move(b_i));
	auto equals_c = std::make_unique<gpgpu::builder::EqualsOperator>(std::move(c_i), std::move(a_plus_b));

	kernel->addComment("Do the operation");
	kernel->addBody(std::move(equals_c));
    	
	// Make sure the user has a compatible device	
	if (builder.hasDevice()) {
		// Helper function that creates a list from [0, a) or [a, b)
        	const auto A = range<int>(100);
		const auto B = range<int>(100, 200);
		std::vector<int> C;

		builder.run("vector_add", &C, A, B);
		// If we want to be explicit about the runtime, we could use:
		// builder.run(gpgpu::Runtime::CUDA, ...)
		// see all options in `include/gpgpu/runtime.hpp`
		process(C);
	} else {
		// If they don't just output the kernel string.
		// Like with builder.run(...), we could also specify a runtime as a parameter and get the text-string representation of our kernel in for that framework.  Does not have to be built with support for that framework to work.
		std::cout << builder.dump() << std::endl;
	}
}
```

## Build Options
The supported build tool is CMake.  The main options necessary are as follows:
```
GPGPU_USE_METAL: (only available Apple, defaults to ON)
GPGPU_USE_CUDA: (only available on Linux/Windows, defaults to ON if CUDA detected)
GPGPU_USE_OPENCL: (only available on Linux/Windows, defaults to ON)
```
Please note that you cannot build CUDA support and Metal support at the same time.  The default settings take this into account.

By `using` a particular framework, this doesn't mean it will necessarily use that framework at runtime.  It just means CMake will build the library with support for that framework.  The runtime framework can be hardcoded if the developer wants to test a particular framework (not recommended outside of testing), or it will be determined based on what OS/graphics card the user has.

## Supported Platforms
Currently being tested on [Travis CI](https://travis-ci.com/github/mattpaletta/gpgpu) on Ubuntu Bionic (CUDA & OpenCL) and MacOS (xcode 12.0).  Because Travis doesn't have GPU-enabled machines (other than MacOS), the tests run, but they do not run the actual GPU code.  The tests just check if it can output the GPU code correctly.  I have CUDA 10.2 on my Windows 10 machine.  I have to test it with CUDA 10.2, because I have a GTX 770, and CUDA 11 doesn't support my card.

GPGPU requires a CXX compiler compatible with C++17, and a CUDA compiler compatible with C++14.

## Dependencies
* [CUDA](https://docs.nvidia.com/cuda/), [CUDART](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html), [NVRTX](https://docs.nvidia.com/cuda/nvrtc/index.html) (if `GPGPU_USE_CUDA`)
* [Jitify](https://github.com/NVIDIA/jitify) (if `GPGPU_USE_CUDA`, header-only library, downloaded through CMake)
* [OpenCL](https://www.khronos.org/opencl/) (if `GPGPU_USE_OPENCL`)
* [Metal](https://developer.apple.com/metal/), [MetalKit](https://developer.apple.com/documentation/metalkit/), [MetalPerformanceShaders](https://developer.apple.com/documentation/metalperformanceshaders/) (if `GPGPU_USE_METAL`)
* [Catch2](https://github.com/catchorg/Catch2) (downloaded and configured through CMake, only required for testing)

## How to use?
```cmake
fetch_extern(gpgpu https://github.com/mattpaletta/gpgpu master)
```
I use FetchContent to along with a helper function to grab this library. You can see that function here: [fetch_extern](https://github.com/mattpaletta/typecheck/blob/master/cmake/fetch_extern.cmake). Alternatively, you can add it as a git submodule include the directory:
```
add_subdirectory(gpgpu)
```

Then you can link it to your project:
```
add_library(my_library ...)
target_link_libraries(my_library PUBLIC ... gpgpu ...)
```

### Similar Projects
* [XShaderCompiler](https://github.com/LukasBanana/XShaderCompiler)
* [CrossShader](https://github.com/alaingalvan/CrossShader)
* [TheVulpes](https://github.com/TheVulpes/TheVulpes)
* [coriander](https://github.com/hughperkins/coriander)

### Usage Note
Only relevant if you enable CUDA: You must have a compatible graphics card for the CUDA version you are using.  I am testing with CUDA 10.2 and 11.0.  With CUDA 11, NVIDIA dropped support for Compute Capability: `3.0, 3.5, and 5.2`.
For more information on GPU products and compute capability, see [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).

## Contribute
I welcome contributions of all sorts.  I consider myself new to the open-source community, so if you're looking for things to contribute, here are some ideas to get started:
- Spelling errors in comments & variable names
- Improve test coverage + add edge cases
- Improvements on performance, readability, etc.
- Suggestions or ideas of larger improvements (leave an issue, and we can discuss)
- Improvements to documentation or code comments to add or update where relevant

## Information

### Questions, Comments, Concerns, Queries, Qwibbles?

If you have any questions, comments, or concerns please leave them in the [GitHub Issues Tracker](https://github.com/mattpaletta/gpgpu/issues)

### Bug reports

If you discover any bugs, feel free to create an issue on GitHub. Please add as much information as possible to help us fixing the possible bug. We also encourage you to help even more by forking and sending us a pull request.

## Maintainers

* Matthew Paletta (https://github.com/mattpaletta)

## License

GPL-3.0 License. Copyright 2020 Matthew Paletta. http://mrated.ca
