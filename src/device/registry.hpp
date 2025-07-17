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
