#include <pybind11/numpy.h>
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

std::unique_ptr<TensorImpl> make_impl_from_shape(const Shape &shape,
                                                 const bool requires_grad,
                                                 const std::string &device,
                                                 const Dtype &dtype) {
  try {
    return registar::TensorImplRegistry::Instance().Create(
        std::nullopt, requires_grad, shape, device, dtype);
  } catch (const std::exception &e) {
    throw pybind11::value_error(std::string("Failed to create tensor impl: ") +
                                e.what());
  }
}

Optional<pybind11::array> object_to_pynp(const Optional<pybind11::object> &data,
                                         const Shape &shape,
                                         const Dtype &dtype) {
  if (data.has_value()) {
    pybind11::module_ np = pybind11::module_::import("numpy");
    pybind11::array np_array = np.attr("array")(
        data.value(), pybind11::arg("dtype") = to_numpy_dtype_string(dtype),
        pybind11::arg("copy") = false);

    if (static_cast<size_t>(np_array.size()) != shape.flat_size()) {
      throw std::runtime_error(
          "Mismatched number of elements between provided data and shape.");
    }

    return std::make_optional(np_array);
  }

  return std::nullopt;
}
} // namespace tensor
} // namespace minigradx