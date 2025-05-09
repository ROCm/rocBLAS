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
# ########################################################################

set(TEMP_CLIENTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../clients)

cmake_minimum_required(VERSION 3.16.8)
project(clients LANGUAGES C CXX)

if(ROCBLAS_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

add_subdirectory(common)

if(ROCBLAS_ENABLE_BENCHMARKS)
    add_subdirectory(bench)
    if(ROCBLAS_BUILD_WITH_TENSILE)
        add_subdirectory(gemm-tune)
    endif()
endif()

if(ROCBLAS_ENABLE_TESTS)
    add_subdirectory(test)
endif()

if(ROCBLAS_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()
