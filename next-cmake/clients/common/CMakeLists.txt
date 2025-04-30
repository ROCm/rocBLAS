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

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/common)

set(rocblas_common_source
    # general
    singletons.cpp
    client_omp.cpp
    client_utility.cpp
    frequency_monitor.cpp
    rocblas_arguments.cpp
    argument_model.cpp
    rocblas_random.cpp
    rocblas_parse_data.cpp
    host_alloc.cpp
    gtest_helpers.cpp
    rocblas_matrix.cpp
)

set(rocblas_common_blas_source
    # general
    cblas_interface.cpp
    # TODO: ADD BLIS SOURCES?
    # blas1
    blas1/common_asum.cpp
    blas1/common_axpy.cpp
    blas1/common_copy.cpp
    blas1/common_dot.cpp
    blas1/common_iamax_iamin.cpp
    blas1/common_nrm2.cpp
    blas1/common_rot.cpp
    blas1/common_scal.cpp
    blas1/common_swap.cpp
    # blas1_ex
    blas_ex/common_axpy_ex.cpp
    blas_ex/common_dot_ex.cpp
    blas_ex/common_nrm2_ex.cpp
    blas_ex/common_rot_ex.cpp
    blas_ex/common_scal_ex.cpp
    # blas2
    blas2/common_trsv.cpp
    blas2/common_gbmv.cpp
    blas2/common_gemv.cpp
    blas2/common_hbmv.cpp
    blas2/common_hemv.cpp
    blas2/common_her.cpp
    blas2/common_her2.cpp
    blas2/common_hpmv.cpp
    blas2/common_hpr.cpp
    blas2/common_hpr2.cpp
    blas2/common_trmv.cpp
    blas2/common_tpmv.cpp
    blas2/common_tbmv.cpp
    blas2/common_tbsv.cpp
    blas2/common_tpsv.cpp
    blas2/common_ger.cpp
    blas2/common_spr.cpp
    blas2/common_spr2.cpp
    blas2/common_syr.cpp
    blas2/common_syr2.cpp
    blas2/common_sbmv.cpp
    blas2/common_spmv.cpp
    blas2/common_symv.cpp
    # blas3 may use tensile or source gemm
    blas3/common_gemm.cpp
    blas_ex/common_gemm_ex.cpp
    blas_ex/common_trsm_ex.cpp
    blas3/common_symm_hemm.cpp
    blas3/common_trsm.cpp
    blas3/common_trtri.cpp
    blas3/common_trmm.cpp
    blas3/common_syrk.cpp
    blas3/common_syr2k.cpp # syrkx
    blas3/common_herk.cpp
    blas3/common_her2k.cpp # herkx
    blas3/common_dgmm.cpp
    blas3/common_geam.cpp
    blas_ex/common_gemmt.cpp
    blas_ex/common_geam_ex.cpp
)

add_library(rocblas-clients-common OBJECT)

# TODO: Is the below if-block needed?
if (WARN_NOT_ILP64_PREFERRED)
  target_compile_definitions(rocblas-clients-common PRIVATE ROCBLAS_REFERENCE_LIB_WARN )
endif()
target_compile_definitions(rocblas-clients-common
    PUBLIC
        ROCBLAS_REFERENCE_LIB=${BLAS_LIBRARY}
        CLIENTS_NO_FORTRAN
        ROCM_USE_FLOAT16
        ROCBLAS_INTERNAL_API
    PRIVATE
        ROCBLAS_NO_DEPRECATED_WARNINGS
        GOOGLE_TEST
        ${TENSILE_DEFINES}  # TODO: Where are these defined?
        ${HIPBLASLT_DEFINES}  # TODO: Where are these defined?
)

target_sources(rocblas-clients-common
    PUBLIC
        ${rocblas_common_source}
        ${rocblas_common_blas_source}
)

target_link_libraries(rocblas-clients-common
    PUBLIC
        roc::rocblas
        hip::host
        hip::device
        BLIS::BLIS
        OpenMP::OpenMP_CXX
        # TODO: Add common link libs? see rocBLAS old build sys
)

target_include_directories(rocblas-clients-common
    PRIVATE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
        # TODO: We don't also need /blas3, /blas2, /blas1?
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex>
        # TODO: ^^^ The above three lines might be able to be added from a helper lib
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
        # TODO: There's a new BLIS integration system, figure that out and add it here
        #       and delete the 4 lines below
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>    # needed?
        $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>    # needed?
        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>    # needed?
        $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>  # needed?

)

target_include_directories(rocblas-clients-common
    PUBLIC
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}>

)
