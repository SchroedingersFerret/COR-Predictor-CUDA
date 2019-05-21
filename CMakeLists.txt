cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(ENV{CUDACXX} "/usr/local/cuda-10.1/bin/nvcc")

project(COR-Predictor-CUDA LANGUAGES CXX CUDA)

include(CTest)

add_library(COR-Predictor-CUDA STATIC
COR-predictor.h
COR-predictor.cu
COR-genetic.h
COR-genetic.cu
COR-anneal.h
COR-anneal.cu
COR-optimization.h
COR-optimization.cu
k-hArray.h
)

target_compile_features(COR-Predictor-CUDA PUBLIC cxx_std_11)

set_target_properties(COR-Predictor-CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(COR-Predictor COR-main.cpp)

set_property(TARGET COR-Predictor PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(COR-Predictor PRIVATE COR-Predictor-CUDA)
