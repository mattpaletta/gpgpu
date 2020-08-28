#include <gpgpu/builder/global_thread.hpp>
#include <gpgpu/builder/constants.hpp>

using namespace gpgpu::builder;

GlobalThread::GlobalThread(std::unique_ptr<BaseBuilder>&& _body) : BaseBuilder(), body(std::move(_body)) {}

std::string GlobalThread::build_opencl(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + "get_global_id(" + this->body->build_opencl(0) + ")";
}

std::string GlobalThread::build_cuda(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + "blockIdx.x * blockDim.x + threadIdx.x";
}

std::string GlobalThread::build_metal(const std::size_t & indentation) const {
    return this->getIndentation(indentation) + constants::metal_id_var_name;
}