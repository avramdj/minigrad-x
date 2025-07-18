#pragma once

#include <cstddef>
#include <vector>
#include <pybind11/pybind11.h>

namespace minigradx {
namespace tensor {

class Shape : public std::vector<int> {
public:
  using std::vector<int>::vector;
  Shape(const std::vector<int>& v);
  size_t flat_size() const;
  static Shape from_data(const pybind11::object &data);
  static Shape null_shape();
};

} // namespace tensor
} // namespace minigradx 