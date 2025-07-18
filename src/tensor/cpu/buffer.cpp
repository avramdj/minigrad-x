#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <tensor/cpu/buffer.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {
namespace cpu {

CpuBuffer::CpuBuffer(size_t size, size_t item_size, Dtype dtype)
    : Buffer(size, item_size, dtype) {
  data = malloc(size * item_size);
  if (!data)
    throw std::bad_alloc();
}
CpuBuffer::CpuBuffer(void *data, size_t size, size_t item_size, Dtype dtype)
    : Buffer(data, size, item_size, dtype) {}

CpuBuffer::~CpuBuffer() {
  if (data != nullptr) {
    free(data);
  }
}

std::shared_ptr<Buffer> CpuBuffer::getitem(const Index &indices) {
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<Buffer> CpuBuffer::clone() const {
  void *new_data = malloc(_size * _item_size);
  if (data != nullptr && new_data != nullptr) {
    memcpy(new_data, data, _size * _item_size);
  }
  return std::make_shared<CpuBuffer>(new_data, _size, _item_size, _dtype);
}

void CpuBuffer::memcpy_from_host(const void *host_data, const size_t &size,
                                 const size_t &item_size) {
  std::memcpy(data, host_data, size * item_size);
}

} // namespace cpu
} // namespace tensor
} // namespace minigradx