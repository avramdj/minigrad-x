#pragma once

#include <pybind11/pybind11.h>
#include <tensor/shape.hpp>
#include <tensor/tensor_impl.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {
namespace cpu {

class CpuImpl : public TensorImpl {
public:
  CpuImpl(const Optional<pybind11::object> &data, const bool requires_grad,
          const Shape &shape, const std::string &device, const Dtype &dtype);
  std::unique_ptr<TensorImpl> getitem(const Index &indices) override;
  static std::unique_ptr<CpuImpl>
  Create(const Optional<pybind11::object> &data, const bool requires_grad,
         const Shape &shape, const std::string &device, const Dtype &dtype);
};

} // namespace cpu
} // namespace tensor
} // namespace minigradx