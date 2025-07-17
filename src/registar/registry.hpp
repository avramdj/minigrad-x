#pragma once

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
