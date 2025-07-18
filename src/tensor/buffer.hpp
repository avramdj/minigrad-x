#pragma once

#include <memory>
#include <tensor/dtype.hpp>
#include <tensor/shape.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {

class Buffer {
protected:
  void *data;
  const size_t _size;
  const size_t _item_size;
  const Dtype _dtype;

  Buffer(size_t size, size_t item_size, Dtype dtype)
      : _size(size), _item_size(item_size), _dtype(dtype) {}
  Buffer(void *data, size_t size, size_t item_size, Dtype dtype)
      : data(data), _size(size), _item_size(item_size), _dtype(dtype) {}

public:
  virtual ~Buffer() = default;
  virtual std::shared_ptr<Buffer> getitem(const Index &indices) = 0;
  virtual void *unsafe_get_data() const { return data; }
  size_t size() const { return _size; }
  size_t item_size() const { return _item_size; }
  virtual std::shared_ptr<Buffer> clone() const = 0;
  Dtype dtype() const { return _dtype; }
  virtual void memcpy_from_host(const void *host_data, const size_t &size,
                                const size_t &item_size) = 0;
};

} // namespace tensor
} // namespace minigradx