#pragma once

#include <tensor/tensor_impl.hpp>

namespace minigradx {
namespace tensor {
namespace cpu {

class CpuImpl : public TensorImpl {
public:
    std::unique_ptr<TensorImpl> getitem(const std::vector<int>& indices) override;
    std::string device() const override;
};

} // namespace cpu
} // namespace tensor
} // namespace minigradx 