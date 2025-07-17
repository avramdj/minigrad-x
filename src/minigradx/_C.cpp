#include "tensor/dtype.hpp"
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
               arg("device"), arg("dtype"));

  ADD_FUNCTION(m, make_impl_from_shape, arg("shape"), arg("requires_grad"),
               arg("device"), arg("dtype"));

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
}