# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(
  rocblas
  VERSION 5.1.0
  LANGUAGES CXX)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Set cpack license file here to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

include(rocblas_target_configure_sanitizers)
include(query_git_hash)
include(FetchROCmCMake)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(GNUInstallDirs)

rocm_setup_version(VERSION ${PROJECT_VERSION})
set(rocblas_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

set(HIP_RUNTIME_MINIMUM
    "4.5.0"
    CACHE STRING "Minimum HIP runtime version required")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(WIN32)
    set(CMAKE_INSTALL_PREFIX
        "C:/hipSDK"
        CACHE PATH "Install path prefix." FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX
        "/opt/rocm"
        CACHE PATH "Install path prefix." FORCE)
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "Export compile_commands.json for clang tooling support")

set(ROCBLAS_ENABLE_TENSILE
    ON
    CACHE BOOL "Build rocBLAS device libraries.")
set(ROCBLAS_ENABLE_CLIENT
    ON
    CACHE BOOL "Build rocBLAS clients.")

if(ROCBLAS_ENABLE_TENSILE)
  if(WIN32)
    set(ROCBLAS_TENSILE_LIBRARY_DIR
        "\${CPACK_PACKAGING_INSTALL_PREFIX}rocblas/bin"
        CACHE PATH "path to tensile library")
  else()
    set(ROCBLAS_TENSILE_LIBRARY_DIR
        "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/rocblas/library"
        CACHE PATH "path to tensile library")
  endif()
  set(ROCBLAS_TENSILE_SUBDIR_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/../../Tensile"
      CACHE STRING "Path to Tensile subdirectory")
endif()

option(BUILD_SHARED_LIBS "Build the rocBLAS shared or static library; master switch." ON)
option(ROCBLAS_BUILD_SHARED_LIBS "Build the rocBLAS shared or static library." ${BUILD_SHARED_LIBS})

option(ROCBLAS_ENABLE_MARKER "Enable rocTracer marker support." OFF)
set(ROCBLAS_ENABLE_OPENMP
    ON
    CACHE BOOL "Enable OpenMP support.")
# I don't know that we can build with BLIS OFF
set(ROCBLAS_ENABLE_BLIS
    ON
    CACHE BOOL "Enable BLIS support.")
set(ROCBLAS_ENABLE_TENSILE
    ON
    CACHE BOOL "Build rocBLAS host library with Tensile backend.")
set(ROCBLAS_ENABLE_HIPBLASLT
    ON
    CACHE BOOL "Build rocBLAS host library with hipBLASLt backend.")
set(ROCBLAS_CONFIG_DIR
    "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Path placed into ldconfig file")
query_git_hash(rocblas ${CMAKE_CURRENT_SOURCE_DIR})


option(BUILD_TESTING "Build test client; master switch." ON)
option(ROCBLAS_BUILD_TESTING "Build test client." ${BUILD_TESTING})

if(ROCBLAS_ENABLE_CLIENT)
  set(ROCBLAS_ENABLE_BENCHMARKS
      ON
      CACHE BOOL "Build benchmark client.")
  set(ROCBLAS_ENABLE_SAMPLES
      ON
      CACHE BOOL "Build client samples.")
  set(ROCBLAS_ENABLE_FORTRAN
      ON
      CACHE BOOL "Build Fortran clients.")
  # rocm-smi is not presently available on Windows so we do not require it.
  cmake_dependent_option(ROCBLAS_ENABLE_ROCM_SMI "Require rocm_smi." ON
                         "NOT WIN32" OFF)
endif()

set(ROCBLAS_ENABLE_ASAN
    OFF
    CACHE BOOL "Build with address sanitizer enabled.")

option(BUILD_CODE_COVERAGE "Build tests with coverage support; master switch." OFF)
option(ROCBLAS_ENABLE_COVERAGE "Build tests with coverage support." ${BUILD_CODE_COVERAGE})

find_package(hip REQUIRED)

if(ROCBLAS_ENABLE_MARKER)
  find_library(ROCTX64_LIBRARY NAMES roctx64 libroctx64)
  if(ROCTX64_LIBRARY)
    message(STATUS "Found roctx64: ${ROCTX64_LIBRARY}")
  else()
    message(WARNING "Could NOT find roctx64: disabling marker support")
    set(ROCBLAS_ENABLE_MARKER OFF)
  endif()
endif()

if(ROCBLAS_ENABLE_BLIS)
  find_package(BLIS REQUIRED)
endif()

if(ROCBLAS_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
  if(NOT TARGET roc::hipblaslt)
    find_package(hipblaslt 1.0.0 REQUIRED)
  endif()
  rocm_package_add_dependencies(DEPENDS "hipblaslt >= 1.0.0")
endif()

if(ROCBLAS_ENABLE_ROCM_SMI)
  find_package(rocm_smi REQUIRED)
endif()

if(ROCBLAS_ENABLE_TENSILE)
  message(STATUS "Configuring device libraries...")
  if(NOT TARGET roc::tensile-host)
    message(STATUS "Adding Tensile from subdirectory: ${ROCBLAS_TENSILE_SUBDIR_PATH}")
    add_subdirectory(${ROCBLAS_TENSILE_SUBDIR_PATH}/next-cmake
                     ${CMAKE_CURRENT_BINARY_DIR}/Tensile)
  endif()
endif()

message(STATUS "Configuring host library")
add_subdirectory(host-library)

if(ROCBLAS_ENABLE_CLIENT)
  message(STATUS "Configuing clients")
  add_subdirectory(clients)
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
  elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION
                                          VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  elseif(CLIENTS_OS STREQUAL "sles")
    set(OPENMP_RPM "libgomp1")
  elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
  endif()

  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
endif()

if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
  rocm_package_setup_client_component(
    tests
    DEPENDS
    COMPONENT
    clients-common
    DEB
    "${OPENMP_DEB}"
    RPM
    "${OPENMP_RPM}")
endif()

if(ROCBLAS_ENABLE_BENCHMARKS)
  rocm_package_setup_client_component(
    benchmarks
    DEPENDS
    COMPONENT
    clients-common
    DEB
    "${OPENMP_DEB}"
    RPM
    "${OPENMP_RPM}")
endif()

if(ROCBLAS_ENABLE_SAMPLES)
  rocm_package_setup_client_component(
    samples
    DEPENDS
    COMPONENT
    clients-common
    DEB
    "${OPENMP_DEB}"
    RPM
    "${OPENMP_RPM}")
endif()

if(ROCBLAS_ENABLE_FORTRAN)
  rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
  rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
  rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS
                                    "${GFORTRAN_RPM}")
  rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS
                                    "${GFORTRAN_DEB}")
endif()

rocm_package_add_deb_dependencies(STATIC_DEPENDS
                                  "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS
                                  "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "\${CPACK_PACKAGING_INSTALL_PREFIX}")

# Work around code object stripping failure if using /usr/bin/strip
set(CPACK_RPM_SPEC_MORE_DEFINE
    "%define __strip ${CMAKE_PREFIX_PATH}/llvm/bin/llvm-strip")

# cmake-format: off
rocm_install(
  TARGETS rocblas
  EXPORT rocblas-targets
  ARCHIVE COMPONENT devel
  LIBRARY COMPONENT runtime
  RUNTIME COMPONENT runtime)

if(ROCBLAS_ENABLE_FORTRAN)
  rocm_install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../library/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas"
    COMPONENT devel
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.f90")
else()
  rocm_install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../library/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas"
    COMPONENT devel
    FILES_MATCHING
      PATTERN "*.h")
endif()

rocm_install(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/rocblas/internal/rocblas-export.h"
  "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/rocblas/internal/rocblas-version.h"
  DESTINATION
  "${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal"
  COMPONENT devel)

# Hack for one off inclusion of files required in the install tree
rocm_install(
  FILES "${PROJECT_SOURCE_DIR}/../library/src/include/rocblas_device_malloc.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal"
  COMPONENT devel)

if(ROCBLAS_ENABLE_OPENMP)
  rocm_export_targets(
    NAMESPACE roc::
    TARGETS rocblas
    EXPORT rocblas-targets
    DEPENDS
      PACKAGE hip
      PACKAGE OpenMP)
else()
  rocm_export_targets(
    NAMESPACE roc::
    TARGETS rocblas
    EXPORT rocblas-targets
    DEPENDS
      PACKAGE hip)
endif()
# cmake-format: on

if(ROCBLAS_ENABLE_TENSILE)
  rocm_install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Tensile/library/"
               DESTINATION "${ROCBLAS_TENSILE_LIBRARY_DIR}" COMPONENT runtime)
endif()

if(ROCBLAS_ENABLE_CLIENT)
  if(ROCBLAS_ENABLE_BENCHMARKS)
    rocm_install(TARGETS rocblas-bench COMPONENT benchmarks DESTINATION
                 ${CMAKE_INSTALL_BINDIR})

    if(ROCBLAS_ENABLE_TENSILE)
      rocm_install(TARGETS rocblas-gemm-tune COMPONENT benchmarks DESTINATION
                   ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()

  if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
    rocm_install(TARGETS rocblas-test COMPONENT tests DESTINATION
                 ${CMAKE_INSTALL_BINDIR})

    if(TARGET rocblas-test-data)
      get_target_property(TEST_DATA rocblas-test-data ROCBLAS_INSTALL_FILES)
      if(TEST_DATA)
        rocm_install(
          FILES ${TEST_DATA}
          DESTINATION "${CMAKE_INSTALL_BINDIR}"
          COMPONENT tests
        )
      endif()
    endif()

    if(TARGET rocblas-test-files)
      get_target_property(TEST_FILES rocblas-test-files ROCBLAS_INSTALL_FILES)
      get_target_property(TEST_PROGRAMS rocblas-test-files ROCBLAS_INSTALL_PROGRAMS)

      if(TEST_FILES)
        rocm_install(
          FILES ${TEST_FILES}
          DESTINATION "${CMAKE_INSTALL_BINDIR}"
          COMPONENT tests
        )
      endif()

      if(TEST_PROGRAMS)
        rocm_install(
          PROGRAMS ${TEST_PROGRAMS}
          DESTINATION "${CMAKE_INSTALL_BINDIR}"
          COMPONENT tests
        )
      endif()
    endif()
  endif()

  if(ROCBLAS_ENABLE_SAMPLES)
    rocm_install(
      TARGETS
      rocblas-example-c-dgeam
      rocblas-example-gemv-graph-capture
      # For whatever reason rocblas-example-header-check isn't installed
      # rocblas-example-header-check
      rocblas-example-hip-complex-her2
      rocblas-example-scal-multiple-strided-batch
      rocblas-example-scal-template
      rocblas-example-sgemm
      rocblas-example-sgemm-multiple-strided-batch
      rocblas-example-sgemm-strided-batched
      rocblas-example-solver
      rocblas-example-sscal
      COMPONENT
      samples
      DESTINATION
      ${CMAKE_INSTALL_BINDIR})

    if(ROCBLAS_ENABLE_TENSILE)
      rocm_install(TARGETS rocblas-example-user-driven-tuning COMPONENT samples
                   DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    if(ROCBLAS_ENABLE_FORTRAN)
      rocm_install(
        TARGETS
        rocblas-example-fortran-axpy
        rocblas-example-fortran-gemv
        rocblas-example-fortran-scal
        COMPONENT
        samples
        DESTINATION
        ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()
endif()

rocm_create_package(
  NAME
  rocblas
  DESCRIPTION
  "BLAS implementation for ROCm, optimized for AMD GPUs."
  MAINTAINER
  "rocBLAS Maintainer <rocblas-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR
  ${ROCBLAS_CONFIG_DIR})

if(ROCBLAS_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE)
  target_compile_options(rocblas PRIVATE -g -O0 -fprofile-instr-generate
                                         -fcoverage-mapping)
  target_link_options(rocblas PRIVATE -fprofile-instr-generate)

  target_compile_options(
    rocblas-clients-common PRIVATE -g -O0 -fprofile-instr-generate
                                   -fcoverage-mapping)
  target_link_options(rocblas-clients-common PRIVATE -fprofile-instr-generate)

  target_compile_options(rocblas-bench PRIVATE -g -O0 -fprofile-instr-generate
                                               -fcoverage-mapping)
  target_link_options(rocblas-bench PRIVATE -fprofile-instr-generate)

  target_compile_options(rocblas-test PRIVATE -g -O0 -fprofile-instr-generate
                                              -fcoverage-mapping)
  target_link_options(rocblas-test PRIVATE -fprofile-instr-generate)

  set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")
  add_custom_target(
    code_cov_tests
    DEPENDS rocblas-test
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
    COMMAND
      ${CMAKE_COMMAND} -E env
      LLVM_PROFILE_FILE="${coverage_dir}/profraw/rocblas-coverage_%p.profraw"
      GTEST_LISTENER=NO_PASS_LINE_IN_LOG $<TARGET_FILE:rocblas-test>
      --precompile=rocblas-test-precompile.db)

  find_program(LLVM_PROFDATA llvm-profdata HINTS llvm/bin REQUIRED)

  find_program(LLVM_COV llvm-cov REQUIRED HINTS llvm/bin)

  add_custom_target(
    coverage
    DEPENDS code_cov_tests
    COMMAND
      ${LLVM_PROFDATA} merge -sparse
      "${coverage_dir}/profraw/rocblas-coverage_*.profraw" -o
      "${coverage_dir}/rocblas.profdata"
    COMMAND ${LLVM_COV} report -object $<TARGET_FILE:rocblas>
            -instr-profile="${coverage_dir}/rocblas.profdata"
    COMMAND
      ${LLVM_COV} show -object $<TARGET_FILE:rocblas>
      -instr-profile="${coverage_dir}/rocblas.profdata" -format=html
      -output-dir="${coverage_dir}")
endif()
