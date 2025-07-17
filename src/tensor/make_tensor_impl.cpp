#include <tensor/make_tensor_impl.hpp>
#include <pybind11/pybind11.h>
#include <registar/registry.hpp>

namespace minigradx {
namespace tensor {

std::unique_ptr<TensorImpl> make_tensor_impl(const pybind11::object& data, const std::string& device) {
    try {
        return registar::TensorImplRegistry::Instance().Create(device);
    } catch (const std::exception& e) {
        throw pybind11::value_error(e.what());
    }
}

} // namespace tensor
} // namespace minigradx