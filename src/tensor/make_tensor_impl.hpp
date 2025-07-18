#pragma once

#include <registar/registry.hpp>
#include <tensor/tensor_impl.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {

std::unique_ptr<TensorImpl>
make_impl_from_data(const Optional<pybind11::object> &data,
                    const bool requires_grad, const std::string &device,
                    const Dtype &dtype);

std::unique_ptr<TensorImpl> make_impl_from_shape(const Shape &shape,
                                                 const bool requires_grad,
                                                 const std::string &device,
                                                 const Dtype &dtype);

Optional<pybind11::array> object_to_pynp(const Optional<pybind11::object> &data,
                                         const Shape &shape,
                                         const Dtype &dtype);

} // namespace tensor
} // namespace minigradx
