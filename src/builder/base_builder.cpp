#include <gpgpu/builder/base_builder.hpp>

using namespace gpgpu::builder;

std::string BaseBuilder::getIndentation(const std::size_t& indentation) const {
    if (indentation == 0) {
        return "";
    }

    std::string out;
    for (int i = 0; i < indentation; ++i) {
        out += "    ";
    }
    return out;
}


bool BaseBuilder::isComment() const { 
    return false; 
}

std::string BaseBuilder::build_opencl(const std::size_t & indentation) const { 
    return ""; 
}

std::string BaseBuilder::build_metal(const std::size_t & indentation) const { 
    return ""; 
}

std::string BaseBuilder::build_cuda(const std::size_t & indentation) const { 
    return ""; 
}
