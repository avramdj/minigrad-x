#pragma once

#include <tensor/tensor_impl.hpp>
#include <registar/registry.hpp>

namespace minigradx {
namespace tensor {
namespace cuda {

class CudaImpl : public TensorImpl {
public:
    std::unique_ptr<TensorImpl> getitem(const std::vector<int>& indices) override;
    std::string device() const override;
};

REGISTER_TENSOR_IMPL("cuda", CudaImpl);

} // namespace cuda
} // namespace tensor
} // namespace minigradx