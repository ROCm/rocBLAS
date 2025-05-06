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

set(ROCBLAS_ENABLE_DEVICE OFF CACHE BOOL "Build rocBLAS device libraries.")
set(ROCBLAS_ENABLE_CLIENT ON CACHE BOOL "Build rocBLAS client apps.")
cmake_dependent_option(ROCBLAS_ENABLE_HOST "Build rocBLAS host library." ON ROCBLAS_ENABLE_CLIENT OFF)

if(ROCBLAS_ENABLE_HOST)
    set(ROCBLAS_BUILD_SHARED ON CACHE BOOL "Build the rocBLAS shared or static library")
    set(ROCBLAS_ENABLE_MSGPACK ON CACHE BOOL "Enable msgpack support.")
    set(ROCBLAS_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP support.")
    set(ROCBLAS_ENABLE_BLIS ON CACHE BOOL "Enable BLIS support.") # I don't know that we can build with this OFF
    set(ROCBLAS_ENABLE_FORTRAN OFF CACHE BOOL "Enable Fortran host and clients.")
    set(ROCBLAS_ENABLE_LLVM OFF CACHE BOOL "Enable LLVM support.")
endif()

if(ROCBLAS_ENABLE_CLIENT)
    # rocm-smi is not presently available on Windows so we do not require it.
    set(ROCBLAS_ENABLE_SAMPLES ON CACHE BOOL "Build rocBLAS client samples.")
    cmake_dependent_option(ROCBLAS_REQUIRE_ROCM_SMI "Require rocm_smi." ON "NOT WIN32" OFF)
endif()

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


# declare targets
# Ideally we keep this top level nice and clean and only add_subdirectory
# here instead of declaring targets

# add subdirectories
if(ROCBLAS_ENABLE_HOST)
    message(STATUS "Building HOST")
    add_subdirectory(host-library)
endif()

if(ROCBLAS_ENABLE_CLIENT)
    # Everything in this block should be contained within a single `add_subdirectory` call.
    message(STATUS "Building CLIENT")
    add_subdirectory(clients)
endif()

# TODO(@ellosal): Uncomment this once the Tensile device libraries are ready
# if(ROCBLAS_ENABLE_DEVICE)
#     message(STATUS "Building DEVICE --- Not implemented yet")
# endif()

# installation

# packaging
