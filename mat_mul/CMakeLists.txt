cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(matrix_mul CUDA CXX)

find_package(OpenMP REQUIRED)

if(NOT DEFINED GEMM_VERSION)
    message(FATAL_ERROR
        "\nGEMM_VERSION not set! Please use one of the following:\n"
        "cmake -DGEMM_VERSION=1D                (for 1D naive GEMM)\n"
        "cmake -DGEMM_VERSION=2D                (for 2D naive GEMM)\n"
        "cmake -DGEMM_VERSION=TILE              (for tiled GEMM)\n"
        "cmake -DGEMM_VERSION=TILE_COARSE_GEMM  (for coarse-grained GEMM)\n"       
    
    ) 
endif()
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(mat_mul mat_mul.cu)
target_link_libraries(mat_mul PUBLIC OpenMP::OpenMP_CXX)

# set_target_properties(mat_mul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(GEMM_VERSION STREQUAL "1D")
    target_compile_definitions(mat_mul PRIVATE NAIVE_GEMM_1D)
    message(STATUS "Building GEMM with 1D configuration")
elseif(GEMM_VERSION STREQUAL "2D")
    target_compile_definitions(mat_mul PRIVATE NAIVE_GEMM_2D)
    message(STATUS "Building GEMM with 2D configuration")
elseif(GEMM_VERSION STREQUAL "TILE")
    target_compile_definitions(mat_mul PRIVATE TILE_GEMM)
    message(STATUS "Building GEMM with tile configuration")
elseif(GEMM_VERSION STREQUAL "TILE_COARSE_GEMM")
    target_compile_definitions(mat_mul PRIVATE TILE_COARSE_GEMM)
    message(STATUS "Building GEMM with coarse tile configuration")
endif()