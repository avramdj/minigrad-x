#include <pybind11/pybind11.h>
#include <registar/registry.hpp>
#include <tensor/make_tensor_impl.hpp>
#include <tensor/shape.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {

std::unique_ptr<TensorImpl>
make_impl_from_data(const Optional<pybind11::object> &data,
                    const bool requires_grad, const std::string &device,
                    const Dtype &dtype) {
  try {
    Shape shape;
    if (data.has_value()) {
      shape = Shape::from_data(data.value());
    } else {
      shape = Shape::null_shape();
    }
    return registar::TensorImplRegistry::Instance().Create(
        data, requires_grad, shape, device, dtype);
  } catch (const std::exception &e) {
    throw pybind11::value_error(std::string("Failed to create tensor impl: ") +
                                e.what());
  }
}

std::unique_ptr<TensorImpl>
make_impl_from_shape(const Shape &shape, const bool requires_grad,
                     const std::string &device, const Dtype &dtype) {
  try {
    return registar::TensorImplRegistry::Instance().Create(
        std::nullopt, requires_grad, shape, device,
        dtype);
  } catch (const std::exception &e) {
    throw pybind11::value_error(std::string("Failed to create tensor impl: ") +
                                e.what());
  }
}

} // namespace tensor
} // namespace minigradx