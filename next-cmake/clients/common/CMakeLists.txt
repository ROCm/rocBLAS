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

find_package(GTest REQUIRED)

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/common)

set(rocblas_client_common_source
    # general
    ${_CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/client_omp.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/client_utility.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/frequency_monitor.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_arguments.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_random.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_parse_data.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/host_alloc.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/gtest_helpers.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_matrix.cpp
)

# Previously called rocblas_testing_common_source
set(rocblas_client_common_blas_source
    # general
    ${_CMAKE_CURRENT_SOURCE_DIR}/cblas_interface.cpp
    # TODO: ADD BLIS SOURCES? Nope, they're added through link libraries
    # blas1
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_asum.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_axpy.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_copy.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_dot.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_iamax_iamin.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_nrm2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_rot.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_scal.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas1/common_swap.cpp
    # blas1_ex
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_axpy_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_dot_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_nrm2_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_rot_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_scal_ex.cpp
    # blas2
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_trsv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_gbmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_gemv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_hbmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_hemv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_her.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_her2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_hpmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_hpr.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_hpr2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_trmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_tpmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_tbmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_tbsv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_tpsv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_ger.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_spr.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_spr2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_syr.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_syr2.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_sbmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_spmv.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas2/common_symv.cpp
    # blas3 may use tensile or source gemm
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_gemm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_gemm_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_trsm_ex.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_symm_hemm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_trsm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_trtri.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_trmm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_syrk.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_syr2k.cpp # syrkx
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_herk.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_her2k.cpp # herkx
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_dgmm.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas3/common_geam.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_gemmt.cpp
    ${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex/common_geam_ex.cpp
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
        # ROCBLAS_NO_DEPRECATED_WARNINGS  # TODO: Why not just let the user handle this?
        BUILD_WITH_TENSILE  # TODO: This should not be unconditionally set on the clients-common target
        ROCBLAS_TENSILE_SEPARATE_ARCH=1
        ROCBLAS_TENSILE_LAZY_LOAD=1
        # ${TENSILE_DEFINES}  # TODO: Where are these defined?
    PRIVATE
        GOOGLE_TEST
        ${HIPBLASLT_DEFINES}  # TODO: Where are these defined?
)

if(ROCBLAS_ENABLE_FORTRAN)
    target_link_libraries(rocblas-clients-common PUBLIC rocblas-fortran-client)
else()
    target_compile_definitions(rocblas-clients-common PUBLIC CLIENTS_NO_FORTRAN)
endif()

target_compile_options(rocblas-clients-common PUBLIC -Wno-unused-result -Wno-deprecated-declarations)

target_sources(rocblas-clients-common
    PRIVATE
        ${rocblas_client_common_source}
        ${rocblas_client_common_blas_source}
)

target_link_libraries(rocblas-clients-common
    PUBLIC
        roc::rocblas
        hip::host
        hip::device
        BLIS::BLIS
        OpenMP::OpenMP_CXX
        Threads::Threads
        GTest::gtest
        rocm_smi64
        # -lm -lstd++fs  # TODO: Are these required?
)

target_include_directories(rocblas-clients-common
    PUBLIC
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}>
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
    # PRIVATE
        # # TODO: There's a new BLIS integration system, figure that out and add it here
        # #       and delete the 4 lines below
        # $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>    # needed?
        # $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>    # needed?
        # $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>    # needed?
        # $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>  # needed?

)

if(ROCBLAS_ENABLE_FORTRAN)
    add_library(rocblas-fortran-client)
    if(ROCBLAS_BUILD_WITH_TENSILE)
        target_sources(rocblas-fortran-client
            PRIVATE
                ${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran_tensile.f90
        )
    endif()
    target_sources(rocblas-fortran-client
        PRIVATE
            ${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran.f90
    )
    target_link_libraries(rocblas-fortran-client PRIVATE rocblas-fortran)
endif()
