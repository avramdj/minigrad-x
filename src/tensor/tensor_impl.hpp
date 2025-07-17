#pragma once

#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <tensor/buffer.hpp>
#include <tensor/dtype.hpp>
#include <tensor/shape.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {

class TensorImpl {
protected:
  std::shared_ptr<Buffer> _buffer;
  std::string _device;
  Dtype _dtype;
  bool _requires_grad;
  Shape _shape;
public:
  virtual ~TensorImpl() = default;
  virtual std::unique_ptr<TensorImpl> getitem(const Index &indices) = 0;
  std::string device() const { return this->_device; }
  Dtype dtype() const { return this->_dtype; }
  bool requires_grad() const { return this->_requires_grad; }
  Shape shape() const { return this->_shape; }
  size_t size() const { return this->_shape.flat_size(); }
};

} // namespace tensor
} // namespace minigradx
