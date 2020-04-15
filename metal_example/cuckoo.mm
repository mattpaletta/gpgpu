#include "cuckoo.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>

#ifdef CUCKOO_SUPPORT_METAL
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#endif

void TestMetal() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    NSString* shadersSrc = [NSString stringWithCString:"#include <metal_stdlib> \n"
                                                       "using namespace metal; \n"
                                                       "kernel void sqr( \n"
                                                       "const device float *vIn [[ buffer(0) ]],\n"
                                                       "    device float *vOut [[ buffer(1) ]],\n"
                                                       "    uint id[[ thread_position_in_grid ]]) {\n"
                                                       "        vOut[id] = vIn[id] * vIn[id];\n"
                                                       "}"
    ];

    std::cout << "Getting library file from file" << std::endl;
//    auto library = [device newLibraryWithSource:shadersSrc options:[ MTLCompileOptions alloc ] error:nullptr];
    auto library = [device newLibraryWithFile:@"/Users/matthew/Projects/cuckoostash/src/CuckooCPU/MyKernels.metallib" error:nullptr];

    std::cout << "Getting function file" << std::endl;
    auto sqrFunc = [library newFunctionWithName:@"sqr"];
    assert(sqrFunc != nullptr);

    auto computePipelineState = [device newComputePipelineStateWithFunction:sqrFunc error:nullptr];
    auto commandQueue = [device newCommandQueue];

    const uint32_t dataCount = 6;

    auto inBuffer = [device newBufferWithLength:sizeof(float) * dataCount options:MTLResourceStorageModeManaged];
    auto outBuffer = [device newBufferWithLength:sizeof(float) * dataCount options:MTLResourceStorageModeManaged];

    for (uint32_t i=0; i<4; i++) {
        // update input data
        auto* inData = static_cast<float*>(inBuffer.contents);
        for (uint32_t j=0; j < dataCount; j++) {
            inData[j] = 10 * i + j;
        }
        [inBuffer didModifyRange: NSMakeRange(0, sizeof(float) * dataCount)];
    }

    auto commandBuffer = [commandQueue commandBuffer];
    auto commandEncoder = [commandBuffer computeCommandEncoder];

    [commandEncoder setBuffer:inBuffer offset:0 atIndex:0];
    [commandEncoder setBuffer:outBuffer offset:0 atIndex:1];
    [commandEncoder setComputePipelineState:computePipelineState];
    [commandEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(dataCount, 1, 1)];
    [commandEncoder endEncoding];

    auto blitCommandEncoder = [commandBuffer blitCommandEncoder];
    [blitCommandEncoder synchronizeResource:outBuffer];
    [blitCommandEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // read the data {
    auto* inData = static_cast<float*>(inBuffer.contents);
    auto* outData = static_cast<float*>(outBuffer.contents);
    for (uint32_t j=0; j<dataCount; j++) {
        printf("sqr(%g) = %g\n", inData[j], outData[j]);
    }
}

int main() {
    TestMetal();
    return 0;
}
