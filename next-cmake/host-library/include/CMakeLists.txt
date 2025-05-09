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

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_HOST_SOURCE_DIR}/include)

configure_file(
    "${_CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-version.h"
    @ONLY
)

generate_export_header(rocblas
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-export.h"
    BASE_NAME rocblas
)

target_sources(rocblas-helper
    INTERFACE
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/check_numerics_matrix.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/check_numerics_vector.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/definitions.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/device_macros.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/handle.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/hipblaslt_host.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/int64_helpers.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/logging.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/macros.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/rocblas_device_malloc.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/rocblas_ostream.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/tensile_host.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/tuple_helper.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../src/include/utility.hpp"
)

target_sources(rocblas
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/rocblas.h"
        "${_CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-types.h"
        "${_CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas_bfloat16.h"
        "${_CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-auxiliary.h"
        "${_CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-functions.h"
        "${_CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-macros.h"
        "${_CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-beta.h"
        "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-version.h"
        "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-export.h"
)

# TODO: add headers in blas1/blas2/blas3/blas_ex directories

add_subdirectory(include64)
