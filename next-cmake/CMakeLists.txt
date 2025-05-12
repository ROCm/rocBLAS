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
project(rocblas VERSION 4.5.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(FetchContent)
include(GenerateExportHeader)
include(GNUInstallDirs)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Export compile_commands.json for clang tooling support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ROCBLAS_ENABLE_DEVICE ON CACHE BOOL "Build rocBLAS device libraries.")
set(ROCBLAS_ENABLE_HOST ON CACHE BOOL "Build rocBLAS host library.")
cmake_dependent_option(ROCBLAS_ENABLE_CLIENT "Build rocBLAS clients." ON ROCBLAS_ENABLE_HOST OFF)

if(ROCBLAS_ENABLE_HOST)
    set(ROCBLAS_BUILD_SHARED ON CACHE BOOL "Build the rocBLAS shared or static library")
    set(ROCBLAS_ENABLE_MSGPACK ON CACHE BOOL "Enable msgpack support.")
    set(ROCBLAS_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP support.")
    set(ROCBLAS_ENABLE_BLIS ON CACHE BOOL "Enable BLIS support.") # I don't know that we can build with this OFF
    set(ROCBLAS_ENABLE_LLVM OFF CACHE BOOL "Enable LLVM support.")
    set(ROCBLAS_ENABLE_TENSILE ON CACHE BOOL "Build rocBLAS host library with Tensile backend.")
    set(ROCBLAS_ENABLE_HIPBLASLT OFF CACHE BOOL "Build rocBLAS host library with hipBLASLt backend (not currently supported).")
endif()

if(ROCBLAS_ENABLE_CLIENT)
    # rocm-smi is not presently available on Windows so we do not require it.
    set(ROCBLAS_ENABLE_BENCHMARKS ON CACHE BOOL "Build benchmark client.")
    set(ROCBLAS_ENABLE_TESTS ON CACHE BOOL "Build benchmark client.")
    set(ROCBLAS_ENABLE_SAMPLES ON CACHE BOOL "Build client samples.")
    set(ROCBLAS_ENABLE_FORTRAN OFF CACHE BOOL "Build Fortran clients.")
    cmake_dependent_option(ROCBLAS_REQUIRE_ROCM_SMI "Require rocm_smi." ON "NOT WIN32" OFF)
endif()


find_package(hip REQUIRED)

if(ROCBLAS_ENABLE_BLIS)
    find_package(BLIS REQUIRED)
endif()

if(ROCBLAS_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(ROCBLAS_ENABLE_LLVM)
    find_package(LLVM REQUIRED)
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
    find_package(roc::hipblaslt REQUIRED)
endif()

if(ROCBLAS_REQUIRE_ROCM_SMI)
    find_package(rocm_smi REQUIRED)
else()
    find_package(rocm_smi)
endif()

FetchContent_Declare(
  rocm-cmake
  GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
  GIT_TAG        develop
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(rocm-cmake)
list(APPEND CMAKE_MODULE_PATH ${rocm-cmake_SOURCE_DIR}/modules)
find_package(ROCmCMakeBuildTools REQUIRED)

# TODO: We want these includes with the others, think of how best to do this...
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)


if(ROCBLAS_ENABLE_HOST)
    message(STATUS "Configuring host library")
    add_subdirectory(host-library)
endif()

if(ROCBLAS_ENABLE_CLIENT)
    find_package(GTest REQUIRED)
    find_package(cblas REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    message(STATUS "Configuing clients")
    add_subdirectory(clients)
endif()

if(ROCBLAS_ENABLE_DEVICE)
    message(STATUS "Configuring device libraries")
    if(NOT TARGET tensile::tensile-host)
        add_subdirectory(Tensile/next-cmake)
    endif()
    if(ROCBLAS_ENABLE_HOST)
        target_link_libraries(rocblas PRIVATE tensile::tensile-host)
    endif()
endif()


rocm_setup_version(VERSION ${PROJECT_VERSION})

# Install host library with proper component segmentation
if(ROCBLAS_ENABLE_HOST)
    rocm_install_targets(
        TARGETS rocblas rocblas64 tensile-host rocblas-helper rocblas-interface
        EXPORT rocblas-targets
        COMPONENT rocblas_runtime
        # INCLUDE DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas"
    )

    # Install public header files
    # TODO: May need to fix the search path and install path of these headers
    rocm_install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.h*"
    )

    # Install generated headers
    rocm_install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/rocblas/internal/rocblas-export.h"
            "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/rocblas/internal/rocblas-version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal"
        COMPONENT devel
    )

    rocm_export_targets(
        NAMESPACE roc::
        TARGETS rocblas rocblas64 tensile-host rocblas-helper
        EXPORT rocblas-targets
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config.cmake"
        VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config-version.cmake"
    )

    rocm_configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocblas-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocblas"
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config-version.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
    )

    rocm_install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocblas"
        COMPONENT rocblas_devel
    )

    rocm_install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
        COMPONENT rocblas_docs
    )
endif()

# Install device library
if(ROCBLAS_ENABLE_DEVICE)
    rocm_install(
        # TODO: I think this directory is wrong
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Tensile/next-cmake/device/library
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocblas
        COMPONENT rocblas_runtime
    )
endif()

# Install client
if(ROCBLAS_ENABLE_CLIENT)
    # TODO: Should I break these into another component? Like `rocblas_client`?
    if(ROCBLAS_ENABLE_BENCHMARKS)
        rocm_install(
            TARGETS rocblas-bench
            EXPORT rocblas-targets
            COMPONENT rocblas_runtime
        )
        if(ROCBLAS_ENABLE_TENSILE)
            rocm_install(
                TARGETS rocblas-gemm-tune
                EXPORT rocblas-targets
                COMPONENT rocblas_runtime
            )
        endif()
    endif()

    if(BUILD_TESTING OR ROCBLAS_ENABLE_TESTS)
        rocm_install(
            TARGETS rocblas-test
            EXPORT rocblas-targets
            COMPONENT rocblas_runtime
        )
    endif()

    if(ROCBLAS_ENABLE_SAMPLES)
        # List all sample targets
        rocm_install(
            TARGETS
                rocblas-example-c-dgeam
                rocblas-example-gemv-graph-capture
                rocblas-example-header-check
                rocblas-example-hip-complex-her2
                rocblas-example-scal-multiple-strided-batch
                rocblas-example-scal-template
                rocblas-example-sgemm
                rocblas-example-sgemm-multiple-strided-batch
                rocblas-example-sgemm-strided-batched
                rocblas-example-solver
                rocblas-example-sscal
                rocblas-example-user-driven-tuning
            EXPORT rocblas-targets
            COMPONENT rocblas_runtime
        )
        if(ROCBLAS_ENABLE_FORTRAN)
            rocm_install(
                TARGETS
                    rocblas-example-fortran-axpy
                    rocblas-example-fortran-gemv
                    rocblas-example-fortran-scal
                EXPORT rocblas-targets
                COMPONENT rocblas_runtime
            )
        endif()
    endif()
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")
set(ROCBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
    NAME rocblas_runtime
    DESCRIPTION "BLAS implementation for ROCm, optimized for AMD GPUs"
    MAINTAINER "rocBLAS Maintainer <rocblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCBLAS_CONFIG_DIR}
)
rocm_create_package(
    NAME rocblas_devel
    DEPENDS COMPONENT rocblas_runtime
    MAINTAINER "rocBLAS Maintainer <rocblas-maintainer@amd.com>"
)
rocm_create_package(
    NAME rocblas_docs
    MAINTAINER "rocBLAS Maintainer <rocblas-maintainer@amd.com>"
)
