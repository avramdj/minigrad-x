#include <cstring>
#include <pybind11/numpy.h>
#include <registar/registry.hpp>
#include <tensor/cpu/buffer.hpp>
#include <tensor/cpu/impl.hpp>
#include <tensor/dtype.hpp>

namespace minigradx {
namespace tensor {
namespace cpu {

std::unique_ptr<TensorImpl> CpuImpl::getitem(const Index &indices) {
  throw std::runtime_error("Not implemented");
}

CpuImpl::CpuImpl(const Optional<pybind11::object> &data,
                 const bool requires_grad, const Shape &shape,
                 const std::string &device, const Dtype &dtype) {

  this->_buffer =
      std::make_shared<CpuBuffer>(shape.flat_size(), DtypeSize(dtype), dtype);

  if (data.has_value()) {
    pybind11::module_ np = pybind11::module_::import("numpy");
    pybind11::array np_array = np.attr("array")(
        data.value(), pybind11::arg("dtype") = to_numpy_dtype_string(dtype),
        pybind11::arg("copy") = false);

    if (static_cast<size_t>(np_array.size()) != shape.flat_size()) {
      throw std::runtime_error(
          "Mismatched number of elements between provided data and shape.");
    }
    std::memcpy(this->_buffer->unsafe_get_data(), np_array.data(),
                shape.flat_size() * DtypeSize(dtype));
  }

  this->_device = device;
  this->_dtype = dtype;
  this->_requires_grad = requires_grad;
  this->_shape = shape;
}

std::unique_ptr<CpuImpl> CpuImpl::Create(const Optional<pybind11::object> &data,
                                         const bool requires_grad,
                                         const Shape &shape,
                                         const std::string &device,
                                         const Dtype &dtype) {
  return std::make_unique<CpuImpl>(data, requires_grad, shape, device, dtype);
}

REGISTER_TENSOR_IMPL(cpu, CpuImpl::Create);

} // namespace cpu
} // namespace tensor
} // namespace minigradx