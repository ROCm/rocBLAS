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

cmake_minimum_required(VERSION 3.25.2)
project(rocblas VERSION 4.5.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(FetchContent)
include(fetch-rocm-cmake)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(GNUInstallDirs)
include(ROCMSetupVersion)
include(ROCMClients)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

rocm_setup_version(VERSION ${PROJECT_VERSION})
set(rocblas_SOVERSION 4.5)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
      set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
    else()
      set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json for clang tooling support")

set(ROCBLAS_ENABLE_DEVICE ON CACHE BOOL "Build rocBLAS device libraries.")
set(ROCBLAS_ENABLE_HOST ON CACHE BOOL "Build rocBLAS host library.")
cmake_dependent_option(ROCBLAS_ENABLE_CLIENT "Build rocBLAS clients." ON ROCBLAS_ENABLE_HOST OFF)

if(ROCBLAS_ENABLE_HOST)
    set(ROCBLAS_BUILD_SHARED ON CACHE BOOL "Build the rocBLAS shared or static library")
    set(ROCBLAS_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP support.")
    set(ROCBLAS_ENABLE_BLIS ON CACHE BOOL "Enable BLIS support.") # I don't know that we can build with this OFF
    set(ROCBLAS_ENABLE_TENSILE ON CACHE BOOL "Build rocBLAS host library with Tensile backend.")
    set(ROCBLAS_ENABLE_HIPBLASLT OFF CACHE BOOL "Build rocBLAS host library with hipBLASLt backend (not currently supported).")
endif()

if(ROCBLAS_ENABLE_CLIENT)
    set(ROCBLAS_ENABLE_BENCHMARKS ON CACHE BOOL "Build benchmark client.")
    set(ROCBLAS_ENABLE_TESTS ON CACHE BOOL "Build benchmark client.")
    set(ROCBLAS_ENABLE_SAMPLES ON CACHE BOOL "Build client samples.")
    set(ROCBLAS_ENABLE_FORTRAN OFF CACHE BOOL "Build Fortran clients.")
    # rocm-smi is not presently available on Windows so we do not require it.
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
    rocm_package_add_dependencies(DEPENDS "hipblaslt >= 0.6.0")
endif()

if(ROCBLAS_REQUIRE_ROCM_SMI)
    find_package(rocm_smi REQUIRED)
else()
    find_package(rocm_smi)
endif()


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
    # TODO(@bstefanuk): Tensile is not currently integrated into rocBLAS
    # if(NOT TARGET tensile::tensile-host)
    #     add_subdirectory(Tensile/next-cmake)
    # endif()
    # if(ROCBLAS_ENABLE_HOST)
    #     target_link_libraries(rocblas PRIVATE tensile::tensile-host)
    # endif()
endif()


if(ROCBLAS_ENABLE_CLIENT)
    if(NOT CLIENTS_OS)
        rocm_set_os_id(CLIENTS_OS)
        string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
        rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
    endif()

    message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")

    set(OPENMP_RPM "libgomp")
    set(OPENMP_DEB "libomp-dev")
    set(GFORTRAN_RPM "libgfortran4")
    set(GFORTRAN_DEB "libgfortran4")

    if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel")
        if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
            set(GFORTRAN_RPM "libgfortran")
        endif()
    elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
        set(GFORTRAN_DEB "libgfortran5")
    elseif(CLIENTS_OS STREQUAL "sles")
        set(OPENMP_RPM "libgomp1")
    elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
        set(GFORTRAN_RPM "gfortran")
    endif()

    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(clients-common)
endif()

if(BUILD_TESTING OR ROCBLAS_ENABLE_TESTS)
    rocm_package_setup_client_component(
        tests
        DEPENDS
            COMPONENT clients-common
            DEB "${OPENMP_DEB}"
            RPM "${OPENMP_RPM}"
    )
endif()

if(ROCBLAS_ENABLE_BENCHMARKS)
    rocm_package_setup_client_component(
        benchmarks
        DEPENDS
            COMPONENT clients-common
            DEB "${OPENMP_DEB}"
            RPM "${OPENMP_RPM}"
    )
endif()

if(ROCBLAS_ENABLE_SAMPLES)
    rocm_package_setup_client_component(
        samples
        DEPENDS
            COMPONENT clients-common
            DEB "${OPENMP_DEB}"
            RPM "${OPENMP_RPM}"
    )
endif()

if(ROCBLAS_ENABLE_FORTRAN)
    rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
    rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
endif()

set(HIP_RUNTIME_MINIMUM 4.5.0)

# TODO: Review when finalizing address sanitizer code
if(BUILD_ADDRESS_SANITIZER)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan")
else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd")
endif()

rocm_package_add_dependencies(SHARED_DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}")

rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT and BSD")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")
# Work around code object stripping failure if using /usr/bin/strip
set(CPACK_RPM_SPEC_MORE_DEFINE "%define __strip /opt/rocm/llvm/bin/llvm-strip")

if(ROCBLAS_ENABLE_HOST)
    rocm_install(
        TARGETS rocblas rocblas64 rocblas-interface
        EXPORT rocblas-targets
        ARCHIVE COMPONENT devel
        LIBRARY COMPONENT runtime
        RUNTIME COMPONENT runtime
    )

    rocm_install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.h*"
    )

    rocm_install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/rocblas/internal/rocblas-export.h"
            "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/rocblas/internal/rocblas-version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal"
        COMPONENT devel
    )

    rocm_export_targets(
        NAMESPACE roc::
        TARGETS rocblas rocblas64 rocblas-interface
        EXPORT rocblas-targets
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config.cmake"
        DEPENDS PACKAGE hip
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
        COMPONENT devel
    )

    rocm_install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
        COMPONENT rocblas_docs
    )
endif()

if(ROCBLAS_ENABLE_DEVICE)
    if(WIN32)
        set(ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}rocblas/bin" CACHE PATH "path to tensile library")
    else()
        set(ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/rocblas" CACHE PATH "path to tensile library")
    endif()
    rocm_install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Tensile/next-cmake/device/library/"
        DESTINATION "${ROCBLAS_TENSILE_LIBRARY_DIR}"
        COMPONENT rocblas_runtime
    )
endif()

if(ROCBLAS_ENABLE_CLIENT)
    if(ROCBLAS_ENABLE_BENCHMARKS)
        rocm_install(
            TARGETS rocblas-bench
            COMPONENT benchmarks
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )

        if(ROCBLAS_ENABLE_TENSILE)
            rocm_install(
                TARGETS rocblas-gemm-tune
                COMPONENT benchmarks
                DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
        endif()
    endif()

    if(ROCBLAS_ENABLE_TESTS)
        rocm_install(
            TARGETS rocblas-test
            COMPONENT tests
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    if(ROCBLAS_ENABLE_SAMPLES)
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
            COMPONENT samples
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )

        if(ROCBLAS_ENABLE_FORTRAN)
            rocm_install(
                TARGETS
                    rocblas-example-fortran-axpy
                    rocblas-example-fortran-gemv
                    rocblas-example-fortran-scal
                COMPONENT samples
                DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
        endif()
    endif()
endif()

set(ROCBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
    NAME rocblas
    DESCRIPTION "BLAS implementation for ROCm, optimized for AMD GPUs."
    MAINTAINER "rocBLAS Maintainer <rocblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCBLAS_CONFIG_DIR}
)
