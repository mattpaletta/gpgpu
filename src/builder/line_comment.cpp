#include <gpgpu/builder/line_comment.hpp>
using namespace gpgpu::builder;

LineComment::LineComment(const std::string& _text) : BaseBuilder(), text(_text) {}

std::string LineComment::build_shared(const std::size_t& indentation) const {
	return this->getIndentation(indentation) + "// " + this->text;
}
			
bool LineComment::isComment() const { 
	return true; 
}

std::string LineComment::build_opencl(const std::size_t & indentation) const { 
	return this->build_shared(indentation); 
}

std::string LineComment::build_metal(const std::size_t & indentation) const { 
	return this->build_shared(indentation); 
}

std::string LineComment::build_cuda(const std::size_t & indentation) const {
	return this->build_shared(indentation); 
}
