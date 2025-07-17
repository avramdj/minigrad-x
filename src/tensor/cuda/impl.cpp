#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <tensor/cuda/buffer.hpp>
#include <tensor/cuda/impl.hpp>
#include <tensor/dtype.hpp>
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

  if (data.has_value()) {
    pybind11::module_ np = pybind11::module_::import("numpy");
    pybind11::array np_array = np.attr("array")(
        data.value(), pybind11::arg("dtype") = to_numpy_dtype_string(dtype),
        pybind11::arg("copy") = false);

    if (static_cast<size_t>(np_array.size()) != shape.flat_size()) {
      throw std::runtime_error(
          "Mismatched number of elements between provided data and shape.");
    }
    cudaMemcpy(this->_buffer->unsafe_get_data(), np_array.data(),
               shape.flat_size() * DtypeSize(dtype), cudaMemcpyHostToDevice);
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