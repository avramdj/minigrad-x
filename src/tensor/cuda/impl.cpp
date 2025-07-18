#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <tensor/cuda/buffer.hpp>
#include <tensor/cuda/impl.hpp>
#include <tensor/dtype.hpp>
#include <tensor/make_tensor_impl.hpp>
#include <tensor/shape.hpp>

namespace minigradx {
namespace tensor {
namespace cuda {

std::unique_ptr<TensorImpl> CudaImpl::getitem(const Index &indices) {
  throw std::runtime_error("Not implemented");
}
CudaImpl::CudaImpl(const Optional<pybind11::object> &data,
                   const bool requires_grad, const Shape &shape,
                   const std::string &device, const Dtype &dtype) {
  this->_buffer =
      std::make_shared<CudaBuffer>(shape.flat_size(), DtypeSize(dtype), dtype);

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

std::unique_ptr<CudaImpl>
CudaImpl::Create(const Optional<pybind11::object> &data,
                 const bool requires_grad, const Shape &shape,
                 const std::string &device, const Dtype &dtype) {
  return std::make_unique<CudaImpl>(data, requires_grad, shape, device, dtype);
}

REGISTER_TENSOR_IMPL(cuda, CudaImpl::Create);

} // namespace cuda
} // namespace tensor
} // namespace minigradx