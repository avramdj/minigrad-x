cmake_minimum_required(VERSION 3.10)
project(minigradx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(third_party/pybind11)

cmake_policy(SET CMP0146 OLD)
find_package(CUDAToolkit QUIET)

file(GLOB_RECURSE SOURCES "src/**/*.cpp")
file(GLOB_RECURSE CUDA_CPP_SOURCES "src/tensor/cuda/*.cpp")
list(REMOVE_ITEM SOURCES ${CUDA_CPP_SOURCES})

set(ALL_SOURCES ${SOURCES})

option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "CUDA support is enabled.")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    list(APPEND ALL_SOURCES ${CUDA_CPP_SOURCES})
    file(GLOB_RECURSE CUDA_SOURCES "src/**/*.cu")
    list(APPEND ALL_SOURCES ${CUDA_SOURCES})
endif()

pybind11_add_module(_C ${ALL_SOURCES})

target_include_directories(_C PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (CUDA_FOUND)
    target_compile_definitions(_C PRIVATE WITH_CUDA)
    target_link_libraries(_C PRIVATE CUDA::cudart)
endif() 