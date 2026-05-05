cmake_minimum_required(VERSION 3.16)
project(algorithm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)

option(ALGO_ENABLE_CUDA_TESTS "Enable CUDA tests when a CUDA compiler is available"
       ON)

if(ALGO_ENABLE_CUDA_TESTS)
  if(NOT CMAKE_CUDA_COMPILER)
    find_program(ALGO_NVCC nvcc HINTS /opt/cuda/bin)
    if(ALGO_NVCC)
      set(CMAKE_CUDA_COMPILER "${ALGO_NVCC}" CACHE FILEPATH "CUDA compiler")
    endif()
  endif()

  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(ALGO_HAS_CUDA ON)
  endif()
endif()

add_library(algo INTERFACE)
target_include_directories(algo INTERFACE lib/include)

enable_testing()
add_subdirectory(test)
add_subdirectory(bench)
add_subdirectory(tools)
