#include <tensor/make_tensor_impl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <minigradx/utils.hpp>

namespace py = pybind11;

using namespace minigradx::tensor;
using arg = py::arg;

#define ADD_CLASS(class_name, scope) \
    py::class_<class_name, std::unique_ptr<class_name>>(scope, #class_name)

#define ADD_METHOD(name, class_name, ...) \
    .def(to_snake_case(#name).c_str(), &class_name::name, ##__VA_ARGS__)

#define ADD_DUNDER_METHOD(name, class_name, ...) \
    .def(("__" + to_snake_case(#name) + "__").c_str(), &class_name::name, ##__VA_ARGS__)

#define ADD_FUNCTION(scope, func, ...) \
    scope.def(#func, func, ##__VA_ARGS__)

#define ADD_READONLY_PROPERTY(name, class_name) \
    .def_property_readonly(#name, &class_name::name)


PYBIND11_MODULE(_C, m) {
    m.doc() = "The minigrad-x C API";

    ADD_CLASS(TensorImpl, m)
    ADD_DUNDER_METHOD(getitem, TensorImpl, arg("indices"))
    ADD_READONLY_PROPERTY(device, TensorImpl)
    ;

    ADD_FUNCTION(m, make_tensor_impl, arg("data"), arg("device"));
} 