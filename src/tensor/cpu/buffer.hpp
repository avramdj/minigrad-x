#pragma once

#include <tensor/buffer.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {
namespace cpu {

class CpuBuffer : public Buffer {
public:
  CpuBuffer(size_t size, size_t item_size, Dtype dtype);
  CpuBuffer(void *data, size_t size, size_t item_size, Dtype dtype);
  ~CpuBuffer() override;
  CpuBuffer(const CpuBuffer &other) = delete;
  CpuBuffer &operator=(const CpuBuffer &other) = delete;
  CpuBuffer(CpuBuffer &&other) = delete;
  CpuBuffer &operator=(CpuBuffer &&other) = delete;
  std::shared_ptr<Buffer> clone() const override;
  std::shared_ptr<Buffer> getitem(const Index &indices) override;
};

} // namespace cpu
} // namespace tensor
} // namespace minigradx