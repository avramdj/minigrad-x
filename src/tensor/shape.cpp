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
} // namespace minigradx