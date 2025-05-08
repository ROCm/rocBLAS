# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ######################################################################### Copyright

cmake_minimum_required(VERSION 3.25.2)  # TODO: Test this with various cmake versions
project(rocblas VERSION 4.5.0 LANGUAGES CXX)  # TODO: Which version?

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CMakeDependentOption)

set(ROCBLAS_ENABLE_DEVICE ON CACHE BOOL "Build rocBLAS device libraries.")
set(ROCBLAS_ENABLE_CLIENT ON CACHE BOOL "Build rocBLAS client apps.")
cmake_dependent_option(ROCBLAS_ENABLE_HOST "Build rocBLAS host library." ON ROCBLAS_ENABLE_CLIENT OFF)

if(ROCBLAS_ENABLE_HOST)
    set(ROCBLAS_BUILD_SHARED ON CACHE BOOL "Build the rocBLAS shared or static library")
    set(ROCBLAS_ENABLE_MSGPACK ON CACHE BOOL "Enable msgpack support.")
    set(ROCBLAS_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP support.")
    set(ROCBLAS_ENABLE_BLIS ON CACHE BOOL "Enable BLIS support.") # I don't know that we can build with this OFF
    set(ROCBLAS_ENABLE_LLVM OFF CACHE BOOL "Enable LLVM support.")
    set(ROCBLAS_BUILD_WITH_TENSILE ON CACHE BOOL "Build rocBLAS host library with Tensile backend.")
    set(ROCBLAS_BUILD_WITH_HIPBLASLT OFF CACHE BOOL "Build rocBLAS host library with hipBLASLt backend (not currently supported).")
endif()

if(ROCBLAS_ENABLE_CLIENT)
    # rocm-smi is not presently available on Windows so we do not require it.
    set(ROCBLAS_ENABLE_BENCHMARKS ON CACHE BOOL "Build benchmark client.")
    set(ROCBLAS_ENABLE_TESTS ON CACHE BOOL "Build benchmark client.")
    set(ROCBLAS_ENABLE_SAMPLES ON CACHE BOOL "Build client samples.")
    set(ROCBLAS_ENABLE_FORTRAN OFF CACHE BOOL "Build Fortran clients.")
    cmake_dependent_option(ROCBLAS_REQUIRE_ROCM_SMI "Require rocm_smi." ON "NOT WIN32" OFF)
endif()

if(ROCBLAS_ENABLE_DEVICE)
    set(ROCBLAS_BUILD_WITH_TENSILE ON CACHE BOOL "Build rocBLAS with Tensile.")
endif()

# Export compile_commands.json for clang tooling support
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

find_package(hip REQUIRED)             # all
find_package(BLIS REQUIRED)            # host & client
find_package(cblas REQUIRED)           # client
find_package(BLAS REQUIRED)            # client?
find_package(LAPACK REQUIRED)          # client?

if(ROCBLAS_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)          # client?
endif()

if(ROCBLAS_ENABLE_MSGPACK)
    # See: https://github.com/msgpack/msgpack-c/wiki/Q%26A#how-to-support-both-msgpack-c-c-version-5x-and-6x-
    # Prefer 6.x (msgpack-cxx) as that is what we bundle in the build.
    find_package(msgpack-cxx CONFIG)
    if(msgpack-cxx_FOUND)
        message(STATUS "Found msgpack-cxx (>=6.x)")
    else()
        find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
        message(STATUS "Found msgpack (<=5.x)")
    endif()
endif()

if(ROCBLAS_ENABLE_LLVM)
    find_package(LLVM REQUIRED)
endif()

if(ROCBLAS_BUILD_WITH_HIPBLASLT)
    find_package(hipblaslt REQUIRED)
endif()

if(ROCBLAS_REQUIRE_ROCM_SMI)
    find_package(rocm_smi REQUIRED)
else()
    find_package(rocm_smi)
endif()

if(ROCBLAS_ENABLE_HOST)
    message(STATUS "Building HOST")
    add_subdirectory(host-library)
endif()

if(ROCBLAS_ENABLE_CLIENT)
    message(STATUS "Building CLIENT")
    add_subdirectory(clients)
endif()

if(ROCBLAS_ENABLE_DEVICE)
    message(STATUS "Building DEVICE with Tensile")
    if(NOT TARGET tensile::tensile-host)
        add_subdirectory(Tensile/next-cmake)
    endif()
    if(ROCBLAS_ENABLE_HOST)
        target_link_libraries(rocblas PRIVATE tensile::tensile-host)
    endif()
endif()

