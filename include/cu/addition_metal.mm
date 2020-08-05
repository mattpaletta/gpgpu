#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <vector>
#include <iostream>

std::vector<float> getMatrix(const std::size_t N) {
    std::vector<float> mat;
    for (std::size_t i = 0; i < N * N; ++i) {
        mat.emplace_back(i);
    }
    return mat;
}

int main() {
    constexpr int N = 100;
    constexpr float alpha = 1.0;
    constexpr float beta = 1.0;

    // Get the metal device and commandQueue to be used.
    auto gDevice = MTLCreateSystemDefaultDevice();
    auto gCommandQueue = gDevice.newCommandQueue;

    auto rowABytes = [MPSMatrixDescriptor rowBytesFromColumns:N dataType:MPSDataTypeFloat32];
    auto matrixADescriptor = [MPSMatrixDescriptor matrixDescriptorWithDimensions:N columns:N rowBytes:rowABytes dataType:MPSDataTypeFloat32];
    auto matrixBDescriptor = [MPSMatrixDescriptor matrixDescriptorWithDimensions:N columns:N rowBytes:rowABytes dataType:MPSDataTypeFloat32];
    auto matrixCDescriptor = [MPSMatrixDescriptor matrixDescriptorWithDimensions:N columns:N rowBytes:rowABytes dataType:MPSDataTypeFloat32];

    const auto AData = getMatrix(N);
    const auto BData = getMatrix(N);

    auto ABuffer = [gDevice newBufferWithBytes:(void *) AData.data() length:N * rowABytes options:MTLResourceStorageModeShared];
    auto BBuffer = [gDevice newBufferWithBytes:(void *) BData.data() length:N * rowABytes options:MTLResourceStorageModeShared];
    auto CBuffer = [gDevice newBufferWithLength:N * rowABytes options:MTLResourceStorageModeShared];

    auto A = [[MPSMatrix alloc] initWithBuffer:ABuffer descriptor:matrixADescriptor];
    auto B = [[MPSMatrix alloc] initWithBuffer:BBuffer descriptor:matrixBDescriptor];
    auto C = [[MPSMatrix alloc] initWithBuffer:CBuffer descriptor:matrixCDescriptor];

    auto sgemmKernal = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice transposeLeft:false transposeRight:false resultRows:N resultColumns:N interiorColumns:N alpha:alpha beta:beta];

    auto commandBuffer = [gCommandQueue commandBuffer];
    [sgemmKernal encodeToCommandBuffer:commandBuffer leftMatrix:A rightMatrix:B resultMatrix:C];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    auto* x = (Float32*) [[C data] contents];

    for (std::size_t i = 0; i < N; ++i) {
        std::cout << "[ ";
        for (std::size_t j = 0; j < N; ++j) {
            std::cout << x[i * N + j] << " ";
        }
        std::cout << "]" << std::endl;
    }

    return 0;
}
