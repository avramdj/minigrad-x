#include <tensor/cuda/cuda_impl.hpp>

namespace minigradx {
namespace tensor {
namespace cuda {


std::unique_ptr<TensorImpl> CudaImpl::getitem(const std::vector<int>& indices) {
    return std::make_unique<CudaImpl>();
}

std::string CudaImpl::device() const {
    return "cuda";
}

} // namespace cuda
} // namespace tensor
} // namespace minigradx 