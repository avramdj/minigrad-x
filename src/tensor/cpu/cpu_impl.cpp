#include <tensor/cpu/cpu_impl.hpp>
#include <registar/registry.hpp>

namespace minigradx {
namespace tensor {
namespace cpu {

std::unique_ptr<TensorImpl> CpuImpl::getitem(const std::vector<int>& indices) {
    return std::make_unique<CpuImpl>();
}

std::string CpuImpl::device() const {
    return "cpu";
}

REGISTER_TENSOR_IMPL("cpu", CpuImpl);


} // namespace cpu
} // namespace tensor
} // namespace minigradx 