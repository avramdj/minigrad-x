#pragma once

#include <registar/registry.hpp>
#include <tensor/tensor_impl.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {
namespace cuda {

class CudaImpl : public TensorImpl {
public:
  CudaImpl(const Optional<pybind11::object> &data, const bool requires_grad,
           const Shape &shape, const std::string &device, const Dtype &dtype);
  std::unique_ptr<TensorImpl> getitem(const Index &indices) override;
  static std::unique_ptr<CudaImpl>
  Create(const Optional<pybind11::object> &data, const bool requires_grad,
         const Shape &shape, const std::string &device, const Dtype &dtype);

};

} // namespace cuda
} // namespace tensor
} // namespace minigradx