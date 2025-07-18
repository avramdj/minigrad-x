#include <cstring>
#include <pybind11/numpy.h>
#include <registar/registry.hpp>
#include <tensor/cpu/buffer.hpp>
#include <tensor/cpu/impl.hpp>
#include <tensor/dtype.hpp>
#include <tensor/make_tensor_impl.hpp>

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

  Optional<pybind11::array> maybe_host_pynp =
      object_to_pynp(data, shape, dtype);

  if (maybe_host_pynp.has_value()) {
    this->_buffer->memcpy_from_host(maybe_host_pynp.value().data(),
                                    shape.flat_size(), DtypeSize(dtype));
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