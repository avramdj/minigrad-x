#pragma once

#include <tensor/tensor_impl.hpp>
#include <unordered_map>
#include <string>
#include <functional>
#include <stdexcept>

using TensorImpl = minigradx::tensor::TensorImpl;

namespace minigradx {
namespace device {

class TensorImplRegistry {
    public:
        using CreatorFn = std::function<std::unique_ptr<TensorImpl>()>;
    
        static TensorImplRegistry& Instance() {
            static TensorImplRegistry instance;
            return instance;
        }
    
        void Register(const std::string& name, CreatorFn creator) {
            if (registry_.count(name)) {
                throw std::runtime_error("TensorImpl already registered for device: " + name);
            }
            registry_[name] = std::move(creator);
        }
    
        std::unique_ptr<TensorImpl> Create(const std::string& name) const {
            auto it = registry_.find(name);
            if (it == registry_.end()) {
                throw std::invalid_argument("Unknown device: " + name);
            }
            return it->second();
        }
    
    private:
        std::unordered_map<std::string, CreatorFn> registry_;
    };

} // namespace device
} // namespace minigradx


#define REGISTER_TENSOR_IMPL(device_name, Type)                           \
    namespace {                                                           \
    struct Type##Registrar {                                              \
        Type##Registrar() {                                               \
            ::minigradx::registar::TensorImplRegistry::Instance().Register( \
                device_name, []() { return std::make_unique<Type>(); });  \
        }                                                                 \
    };                                                                    \
    static Type##Registrar global_##Type##Registrar_instance;             \
    }
#pragma once

#include <string>

std::string to_snake_case(const std::string& input);#include "tensor/dtype.hpp"
#include <minigradx/utils.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tensor/make_tensor_impl.hpp>

namespace py = pybind11;

using namespace minigradx::tensor;
using arg = py::arg;

#define ADD_CLASS(class_name, scope)                                           \
  py::class_<class_name, std::unique_ptr<class_name>>(scope, #class_name)

#define ADD_METHOD(name, class_name, ...)                                      \
  .def(to_snake_case(#name).c_str(), &class_name::name, ##__VA_ARGS__)

#define ADD_DUNDER_METHOD(name, class_name, ...)                               \
  .def(("__" + to_snake_case(#name) + "__").c_str(), &class_name::name,        \
       ##__VA_ARGS__)

#define ADD_FUNCTION(scope, func, ...) scope.def(#func, func, ##__VA_ARGS__)

#define ADD_READONLY_PROPERTY(name, class_name)                                \
  .def_property_readonly(#name, &class_name::name)

/////////    Actual code    ///////////
PYBIND11_MODULE(_C, m) {
  m.doc() = "The minigrad-x C API";

  {
    ADD_CLASS(TensorImpl, m)
    ADD_DUNDER_METHOD(getitem, TensorImpl, arg("indices"))
    ADD_READONLY_PROPERTY(device, TensorImpl);
  }

  ADD_FUNCTION(m, make_impl_from_data, arg("data"), arg("requires_grad"),
               arg("shape"), arg("device"), arg("dtype"));

  py::enum_<Dtype>(m, "Dtype")
      .value("Float64", Dtype::Float64)
      .value("Float32", Dtype::Float32)
      .value("Float16", Dtype::Float16)
      .value("Int64", Dtype::Int64)
      .value("Int32", Dtype::Int32)
      .value("Int16", Dtype::Int16)
      .value("Int8", Dtype::Int8)
      .value("UInt64", Dtype::UInt64)
      .value("UInt32", Dtype::UInt32)
      .value("UInt16", Dtype::UInt16)
      .value("UInt8", Dtype::UInt8)
      .value("Bool", Dtype::Bool)
      .export_values();
}#include <minigradx/utils.hpp>
#include <cctype>

std::string to_snake_case(const std::string& input) {
    std::string out;
    out.reserve(input.size() * 2);

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (std::isupper(static_cast<unsigned char>(c))) {
            if (i > 0) {
                out.push_back('_');
            }
            out.push_back(static_cast<char>(std::tolower(c)));
        } else {
            out.push_back(c);
        }
    }
    return out;
}#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <tensor/tensor_impl.hpp>
#include <tensor/utils.hpp>
#include <unordered_map>
#include <tensor/shape.hpp>

using TensorImpl = minigradx::tensor::TensorImpl;

namespace minigradx {
namespace registar {

class TensorImplRegistry {
public:
  using CreatorFn = std::function<std::unique_ptr<TensorImpl>(
      const Optional<pybind11::object> &data, const bool requires_grad,
      const tensor::Shape &shape, const std::string &device,
      const tensor::Dtype &dtype)>;

  static TensorImplRegistry &Instance() {
    static TensorImplRegistry instance;
    return instance;
  }

  void Register(const std::string &name, CreatorFn creator) {
    if (registry_.count(name)) {
      throw std::runtime_error("TensorImpl already registered for device: " +
                               name);
    }
    registry_[name] = std::move(creator);
  }

  std::unique_ptr<TensorImpl> Create(const Optional<pybind11::object> &data,
                                     const bool requires_grad,
                                     const tensor::Shape &shape,
                                     const std::string &device,
                                     const tensor::Dtype &dtype) const {
    return creator_fn_(device)(data, requires_grad, shape, device, dtype);
  }

private:
  std::unordered_map<std::string, CreatorFn> registry_;

  CreatorFn creator_fn_(const std::string &device_name) const {
    auto it = registry_.find(device_name);
    if (it == registry_.end()) {
      throw std::invalid_argument("Unknown device: " + device_name);
    }
    return it->second;
  }
};

} // namespace registar
} // namespace minigradx

#define REGISTER_TENSOR_IMPL(device_name, creator_fn)                          \
  namespace {                                                                  \
  struct device_name##Registrar {                                              \
    device_name##Registrar() {                                                 \
      ::minigradx::registar::TensorImplRegistry::Instance().Register(          \
          std::string(#device_name), creator_fn);                              \
    }                                                                          \
  };                                                                           \
  static device_name##Registrar global_##device_name##Registrar_instance;      \
  }
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
} // namespace minigradx#pragma once

#include <registar/registry.hpp>
#include <tensor/tensor_impl.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {
namespace cuda {

class CudaImpl : public TensorImpl {
public:
  CudaImpl(const Optional<pybind11::object> &data, const bool requires_grad,
           const Shape &shape, const std::string &device, const Dtype &dtype);
  std::unique_ptr<TensorImpl> getitem(const Index &indices) override;
  static std::unique_ptr<CudaImpl>
  Create(const Optional<pybind11::object> &data, const bool requires_grad,
         const Shape &shape, const std::string &device, const Dtype &dtype);

};

} // namespace cuda
} // namespace tensor
} // namespace minigradx#include <cuda_runtime.h>
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
} // namespace minigradx#include <tensor/cuda/buffer.hpp>
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
} // namespace minigradx#pragma once

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
} // namespace minigradx#pragma once

#include <pybind11/pybind11.h>
#include <tensor/shape.hpp>
#include <tensor/tensor_impl.hpp>
#include <tensor/utils.hpp>

namespace minigradx {
namespace tensor {
namespace cpu {

class CpuImpl : public TensorImpl {
public:
  CpuImpl(const Optional<pybind11::object> &data, const bool requires_grad,
          const Shape &shape, const std::string &device, const Dtype &dtype);
  std::unique_ptr<TensorImpl> getitem(const Index &indices) override;
  static std::unique_ptr<CpuImpl>
  Create(const Optional<pybind11::object> &data, const bool requires_grad,
         const Shape &shape, const std::string &device, const Dtype &dtype);
};

} // namespace cpu
} // namespace tensor
} // namespace minigradx#include <cstdlib>
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

} // namespace cpu
} // namespace tensor
} // namespace minigradx#include <registar/registry.hpp>
#include <tensor/cpu/buffer.hpp>
#include <tensor/cpu/impl.hpp>

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
} // namespace minigradx#pragma once

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
} // namespace minigradx#pragma once

#include <optional>
#include <vector>

template<typename T>
using Optional = std::optional<T>;

namespace minigradx {
namespace tensor {

using Index = std::vector<int>;

} // namespace tensor
} // namespace minigradx
#include <tensor/shape.hpp>

namespace minigradx {
namespace tensor {

size_t Shape::flat_size() const {
  size_t size = 1;
  for (int dim : *this) {
    size *= dim;
  }
  return size;
}

Shape Shape::from_data(const pybind11::object &data) {
  namespace py = pybind11;

  // has .shape
  if (py::hasattr(data, "shape")) {
    auto py_shape = data.attr("shape").cast<py::tuple>();
    Shape s;
    for (auto &dim : py_shape) {
      s.push_back(dim.cast<int>());
    }
    return s;
  }

  // Python sequence
  if (py::isinstance<py::sequence>(data)) {
    auto seq = data.cast<py::sequence>();
    ssize_t n = seq.size();
    Shape s;
    s.push_back(int(n));

    if (n > 0) {
      // recurse on first element
      Shape sub = Shape::from_data(seq[0]);
      s.insert(s.end(), sub.begin(), sub.end());

      // verify all elements agree
      for (ssize_t i = 1; i < n; ++i) {
        Shape other = Shape::from_data(seq[i]);
        if (other != sub) {
          throw std::runtime_error("Inconsistent shapes at index " +
                                   std::to_string(i));
        }
      }
    }
    return s;
  }

  // 3) scalar
  return Shape::null_shape();
}

Shape Shape::null_shape() { return Shape({}); }

} // namespace tensor
} // namespace minigradx#pragma once

#include <stdexcept>

namespace minigradx {
namespace tensor {

enum class Dtype {
    Float64,
    Float32,
    Float16,
    Int64,
    Int32,
    Int16,
    Int8,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
    Bool,
};

static constexpr unsigned int DtypeSize(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float64: return 8;
        case Dtype::Float32: return 4;
        case Dtype::Float16: return 2;
        case Dtype::Int64: return 8;
        case Dtype::Int32: return 4;
        case Dtype::Int16: return 2;
        case Dtype::Int8: return 1;
        case Dtype::UInt64: return 8;
        case Dtype::UInt32: return 4;
        case Dtype::UInt16: return 2;
        case Dtype::UInt8: return 1;
        case Dtype::Bool: return 1;
        default: throw std::invalid_argument("Invalid dtype");
    }
};

} // namespace tensor
} // namespace minigradx#pragma once

#include <registar/registry.hpp>
#include <tensor/tensor_impl.hpp>

namespace minigradx {
namespace tensor {

std::unique_ptr<TensorImpl>
make_impl_from_data(const Optional<pybind11::object> &data,
                    const bool requires_grad, const std::string &device,
                    const Dtype &dtype);

std::unique_ptr<TensorImpl>
make_impl_from_shape(const Optional<pybind11::object> &data, 
                    const bool requires_grad, const std::string &device,
                    const Dtype &dtype);

} // namespace tensor
} // namespace minigradx
#include <pybind11/pybind11.h>
#include <registar/registry.hpp>
#include <tensor/make_tensor_impl.hpp>
#include <tensor/shape.hpp>

namespace minigradx {
namespace tensor {

std::unique_ptr<TensorImpl>
make_impl_from_data(const Optional<pybind11::object> &data,
                    const bool requires_grad, const std::string &device,
                    const Dtype &dtype) {
  try {
    Shape shape;
    if (data.has_value()) {
      shape = Shape::from_data(data.value());
    } else {
      shape = Shape::null_shape();
    }
    return registar::TensorImplRegistry::Instance().Create(
        data, requires_grad, shape, device, dtype);
  } catch (const std::exception &e) {
    throw pybind11::value_error(std::string("Failed to create tensor impl: ") +
                                e.what());
  }
}

} // namespace tensor
} // namespace minigradx#pragma once

#include <memory>
#include <tensor/dtype.hpp>
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
  size_t size() const { return _size; }
  size_t item_size() const { return _item_size; }
  virtual std::shared_ptr<Buffer> clone() const = 0;
  Dtype dtype() const { return _dtype; }
};

} // namespace tensor
} // namespace minigradx#pragma once

#include <cstddef>
#include <vector>
#include <pybind11/pybind11.h>

namespace minigradx {
namespace tensor {

class Shape : public std::vector<int> {
public:
  using std::vector<int>::vector;
  size_t flat_size() const;
  static Shape from_data(const pybind11::object &data);
  static Shape null_shape();
};

} // namespace tensor
} // namespace minigradx from .tensor.tensor import Tensor

__all__ = ["Tensor"]
from __future__ import annotations

from typing import Optional

from minigradx._C import Dtype, TensorImpl, make_impl_from_data
from minigradx.tensor._types import ConvertableToTensor


class Tensor:
    def __init__(
        self,
        data: Optional[ConvertableToTensor] = None,
        device: str = "cpu",
        dtype: Dtype = Dtype.Float32,
        requires_grad: bool = True,
    ) -> None:
        self._impl = make_impl_from_data(
            data=data,
            requires_grad=requires_grad,
            device=device,
            dtype=dtype,
        )

    def __getitem__(self, indices: tuple[int, ...] | list[int]) -> Tensor:
        return self._from_impl(self._impl.__getitem__(indices))

    @property
    def device(self) -> str:
        return self._impl.device

    # @property
    # def grad(self) -> Optional[Tensor]:
    #     if self._impl.grad is None:
    #         return None
    #     return Tensor._from_impl(self._impl.grad)

    # @property
    # def shape(self) -> tuple[int, ...]:
    #     return self._impl.shape

    @classmethod
    def _from_impl(cls, impl: TensorImpl, copy: bool = False) -> Tensor:
        tensor = cls(device=impl.device)
        if copy:
            tensor._impl = impl.clone()
        else:
            tensor._impl = impl

        return tensor
import typing

Numeric = typing.Union[int, float, bool]

NestedNumeric = typing.Union[
    Numeric, list["NestedNumeric"], tuple["NestedNumeric", ...]
]

ConvertableToTensor = typing.Union[NestedNumeric]
