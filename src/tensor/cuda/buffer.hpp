#pragma once

#include <tensor/buffer.hpp>
#include <tensor/dtype.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {
namespace cuda {

class CudaBuffer : public Buffer {
public:
  CudaBuffer(size_t size, size_t item_size, const Dtype &dtype);
  CudaBuffer(void *data, size_t size, size_t item_size, const Dtype &dtype);
  ~CudaBuffer() override;
  CudaBuffer(const CudaBuffer &other) = delete;
  CudaBuffer &operator=(const CudaBuffer &other) = delete;
  CudaBuffer(CudaBuffer &&other) = delete;
  CudaBuffer &operator=(CudaBuffer &&other) = delete;
  std::shared_ptr<Buffer> clone() const override;
  std::shared_ptr<Buffer> getitem(const Index &indices) override;
};

} // namespace cuda
} // namespace tensor
} // namespace minigradx