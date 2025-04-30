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

set(TEMP_HOST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../library)
set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_HOST_SOURCE_DIR})

cmake_minimum_required(VERSION 3.25.2)
project(host-library VERSION 4.5.0 LANGUAGES CXX)

add_library(rocblas)
add_library(roc::rocblas ALIAS rocblas)

# TODO: goes into `host-library/include`
configure_file(
    ${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal/rocblas-version.h
    @ONLY
)

generate_export_header(rocblas
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal/rocblas-export.h
    BASE_NAME rocblas
)

set(rocblas_headers_public
)

target_sources(rocblas
    PRIVATE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/rocblas.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-types.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas_bfloat16.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-auxiliary.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-functions.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-macros.h>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-beta.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal/rocblas-version.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal/rocblas-export.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/rocblas.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/internal/rocblas-types.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/internal/rocblas_bfloat16.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/internal/rocblas-auxiliary.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/internal/rocblas-functions.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/internal/rocblas-macros.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/internal/rocblas-beta.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/rocblas/internal/rocblas-version.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/rocblas/internal/rocblas-export.h>
)


target_include_directories(rocblas
    PRIVATE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src/include>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src/src64>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas>  # TODO: Required?
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(rocblas
    PRIVATE
        hip::host
        hip::device
        OpenMP::OpenMP_CXX
)
if(ROCBLAS_BUILD_WITH_HIPBLASLT)
    target_link_libraries(rocblas
        PRIVATE
            roc::hipblaslt
    )
endif()

target_compile_features(rocblas PRIVATE cxx_std_17)

target_compile_definitions(rocblas
    PRIVATE
        ROCM_USE_FLOAT16
        ROCBLAS_INTERNAL_API
        # ROCBLAS_INTERNAL_ILP64
        ROCBLAS_BETA_FEATURES_API
        rocblas_EXPORTS # TODO: needed?
)

set_target_properties(rocblas
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/staging
)

# add_subdirectory(include)  # TODO: empty, but needed
add_subdirectory(src)
