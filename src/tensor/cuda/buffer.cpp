#include <cuda_runtime.h>
#include <tensor/cuda/buffer.hpp>
#include <tensor/dtype.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {
namespace cuda {

CudaBuffer::CudaBuffer(size_t size, size_t item_size, const Dtype &dtype)
    : Buffer(size, item_size, dtype) {
  cudaMalloc(&data, size * item_size);
  if (!data)
    throw std::bad_alloc();
}

CudaBuffer::CudaBuffer(void *data, size_t size, size_t item_size,
                       const Dtype &dtype)
    : Buffer(data, size, item_size, dtype) {}

CudaBuffer::~CudaBuffer() {
  if (data != nullptr) {
    cudaFree(data);
  }
}

std::shared_ptr<Buffer> CudaBuffer::getitem(const Index &indices) {
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<Buffer> CudaBuffer::clone() const {
  void *new_data;
  cudaMalloc(&new_data, _size * _item_size);
  if (data != nullptr && new_data != nullptr) {
    cudaMemcpy(new_data, data, _size * _item_size, cudaMemcpyDeviceToDevice);
  }
  return std::make_shared<CudaBuffer>(new_data, _size, _item_size, _dtype);
}

} // namespace cuda
} // namespace tensor
} // namespace minigradx