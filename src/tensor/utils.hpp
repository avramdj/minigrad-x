#pragma once

#include <optional>
#include <vector>

template<typename T>
using Optional = std::optional<T>;

namespace minigradx {
namespace tensor {

using Index = std::vector<int>;

} // namespace tensor
} // namespace minigradx
