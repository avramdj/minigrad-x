#pragma once

#include <cstdint>
#include <string>
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

inline unsigned int DtypeSize(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float64: return 8;
        case Dtype::Float32: return 4;
        case Dtype::Float16: return 2;
        case Dtype::Int64:   return 8;
        case Dtype::Int32:   return 4;
        case Dtype::Int16:   return 2;
        case Dtype::Int8:    return 1;
        case Dtype::UInt64:  return 8;
        case Dtype::UInt32:  return 4;
        case Dtype::UInt16:  return 2;
        case Dtype::UInt8:   return 1;
        case Dtype::Bool:    return 1;
        default: throw std::invalid_argument("Invalid dtype");
    }
};

inline std::string to_numpy_dtype_string(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float64: return "float64";
        case Dtype::Float32: return "float32";
        case Dtype::Float16: return "float16";
        case Dtype::Int64:   return "int64";
        case Dtype::Int32:   return "int32";
        case Dtype::Int16:   return "int16";
        case Dtype::Int8:    return "int8";
        case Dtype::UInt64:  return "uint64";
        case Dtype::UInt32:  return "uint32";
        case Dtype::UInt16:  return "uint16";
        case Dtype::UInt8:   return "uint8";
        case Dtype::Bool:    return "bool";
        default: throw std::invalid_argument("Unsupported dtype for numpy conversion");
    }
}

} // namespace tensor
} // namespace minigradx