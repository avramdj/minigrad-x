#pragma once

#include <pybind11/pybind11.h>
#include <string>
#include <memory>
#include <vector>

namespace minigradx {
namespace tensor {

class TensorImpl {
public:
    virtual ~TensorImpl() = default;
    virtual std::unique_ptr<TensorImpl> getitem(const std::vector<int>& indices) = 0;
    virtual std::string device() const = 0;
};

} // namespace tensor
} // namespace minigradx 