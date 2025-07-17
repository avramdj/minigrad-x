#pragma once

#include <registar/registry.hpp>
#include <tensor/tensor_impl.hpp>

namespace minigradx {
namespace tensor {

std::unique_ptr<TensorImpl>
make_impl_from_data(const Optional<pybind11::object> &data,
                    const bool requires_grad, const std::string &device,
                    const Dtype &dtype);

std::unique_ptr<TensorImpl>
make_impl_from_shape(const Shape &shape, const bool requires_grad,
                     const std::string &device, const Dtype &dtype);

} // namespace tensor
} // namespace minigradx
