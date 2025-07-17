#pragma once

#include <tensor/tensor_impl.hpp>

namespace minigradx {
namespace tensor {

std::unique_ptr<TensorImpl> make_tensor_impl(const pybind11::object& data, const std::string& device);

} // namespace tensor
} // namespace minigradx
