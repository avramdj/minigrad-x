cmake_minimum_required(VERSION 3.10)
project(minigradx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(third_party/pybind11)

find_package(CUDAToolkit)

file(GLOB_RECURSE SOURCES "src/**/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/**/*.cu")

pybind11_add_module(_C ${SOURCES} ${CUDA_SOURCES})

target_include_directories(_C PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(_C PRIVATE CUDA::cudart)

if (CUDA_FOUND)
    target_link_libraries(_C PRIVATE CUDA::cudart)
endif() 