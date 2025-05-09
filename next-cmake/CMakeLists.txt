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

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CMakeDependentOption)


set(ROCBLAS_ENABLE_DEVICE ON CACHE BOOL "Build rocBLAS device libraries.")
set(ROCBLAS_ENABLE_HOST ON CACHE BOOL "Build rocBLAS host library.")
cmake_dependent_option(ROCBLAS_ENABLE_CLIENT "Build rocBLAS clients." ON ROCBLAS_ENABLE_HOST OFF)

if(ROCBLAS_ENABLE_HOST)
    set(ROCBLAS_BUILD_SHARED ON CACHE BOOL "Build the rocBLAS shared or static library")
    set(ROCBLAS_ENABLE_MSGPACK ON CACHE BOOL "Enable msgpack support.")
    set(ROCBLAS_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP support.")
    set(ROCBLAS_ENABLE_BLIS ON CACHE BOOL "Enable BLIS support.") # I don't know that we can build with this OFF
    set(ROCBLAS_ENABLE_LLVM OFF CACHE BOOL "Enable LLVM support.")
    # TODO: Use ENABLE_ prefix for these options
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

# Export compile_commands.json for clang tooling support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


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

if(ROCBLAS_BUILD_WITH_HIPBLASLT)
    find_package(roc::hipblaslt REQUIRED)
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
    if(NOT TARGET tensile::tensile-host)
        add_subdirectory(Tensile/next-cmake)
    endif()
    if(ROCBLAS_ENABLE_HOST)
        target_link_libraries(rocblas PRIVATE tensile::tensile-host)
    endif()
endif()

# if(ROCBLAS_ENABLE_HOST)
#     install(TARGETS rocblas
#         EXPORT rocblas-targets
#         LIBRARY
#             DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#             COMPONENT rocBLAS_Runtime
#         ARCHIVE
#             DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#             COMPONENT rocBLAS_Development
#         RUNTIME
#             DESTINATION "${CMAKE_INSTALL_BINDIR}"
#             COMPONENT rocBLAS_Runtime
#         PUBLIC_HEADER
#             DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
#             COMPONENT rocBLAS_Development
#         INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
#     )
#     install(
#         EXPORT rocblas-targets
#         NAMESPACE roc::
#         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocblas
#     )
#     get_target_property(rocblas_public_headers roc::rocblas INTERFACE_SOURCES)
#     install(
#         FILES ${rocblas_public_headers}
#         DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas"
#         COMPONENT rocBLAS_Development
#     )

#     install(
#         FILES
#             "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/rocblas/internal/rocblas-export.h"
#             "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/rocblas/internal/rocblas-version.h"
#         DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
#         COMPONENT rocBLAS_Development
#     )

#     write_basic_package_version_file(
#         "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config-version.cmake"
#         VERSION ${PROJECT_VERSION}
#         COMPATIBILITY SameMajorVersion
#     )

#     configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocblas-config.cmake.in"
#         "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config.cmake" @ONLY)
#     install(FILES
#         "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config-version.cmake"
#         "${CMAKE_CURRENT_BINARY_DIR}/rocblas-config.cmake"
#         DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocblas"
#         COMPONENT rocBLAS_Development
#     )

#     install(
#         FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md"
#         DESTINATION "${CMAKE_INSTALL_DOCDIR}"
#         COMPONENT rocBLAS_Docs
#     )

#     if(ROCBLAS_ENABLE_DEVICE)
#         install(
#             DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Tensile/next-cmake/device/library"
#             DESTINATION "${CMAKE_INSTALL_LIBDIR}/rocblas"
#             COMPONENT rocBLAS_Runtime
#         )
#     endif()

#     if(ROCBLAS_ENABLE_CLIENT)
#         if(ROCBLAS_ENABLE_BENCHMARKS)
#             install(
#                 TARGETS rocblas-bench
#                 EXPORT rocblas-targets
#                 COMPONENT rocBLAS_Runtime
#             )
#             if(ROCBLAS_BUILD_WITH_TENSILE)
#                 install(
#                     TARGETS rocblas-gemm-tune
#                     EXPORT rocblas-targets
#                     COMPONENT rocBLAS_Runtime
#                 )
#             endif()
#         endif()

#         if(BUILD_TESTING OR ROCBLAS_ENABLE_TESTS)
#             install(
#                 TARGETS rocblas-test
#                 EXPORT rocblas-targets
#                 COMPONENT rocBLAS_Runtime
#             )
#             install(
#                 FILES "${CMAKE_CURRENT_BINARY_DIR}/clients/rocblas_gtest.data"
#                 DESTINATION "${CMAKE_INSTALL_BINDIR}"
#                 COMPONENT rocBLAS_Runtime
#             )
#         endif()

#         if(ROCBLAS_ENABLE_SAMPLES)
#             # List all sample targets
#             install(
#                 TARGETS
#                     rocblas-example-c-dgeam
#                     rocblas-example-gemv-graph-capture
#                     rocblas-example-header-check
#                     rocblas-example-hip-complex-her2
#                     rocblas-example-scal-multiple-strided-batch
#                     rocblas-example-scal-template
#                     rocblas-example-sgemm
#                     rocblas-example-sgemm-multiple-strided-batch
#                     rocblas-example-sgemm-strided-batched
#                     rocblas-example-solver
#                     rocblas-example-sscal
#                     rocblas-example-user-driven-tuning
#                 EXPORT rocblas-targets
#                 COMPONENT rocBLAS_Runtime
#             )
#             if(ROCBLAS_ENABLE_FORTRAN)
#                 install(
#                     TARGETS
#                         rocblas-example-fortran-axpy
#                         rocblas-example-fortran-gemv
#                         rocblas-example-fortran-scal
#                     EXPORT rocblas-targets
#                     COMPONENT rocBLAS_Runtime
#                 )
#             endif()
#         endif()
#     endif()

#     set(CPACK_PACKAGE_NAME "rocBLAS")
#     set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
#     set(CPACK_PACKAGE_CONTACT "rocBLAS Maintainer <rocblas-maintainer@amd.com>")
#     set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ROCm BLAS marshalling library")
#     set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
#     set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
#     set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
#     set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
#     set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_LIST_DIR}/../LICENSE.md)
#     set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_LIST_DIR}/../README.md)
#     set(CPACK_VERBATIM_VARIABLES YES)
#     set(CPACK_COMPONENTS_ALL "rocBLAS_Runtime;rocBLAS_Development;rocBLAS_Docs")

#     set(CPACK_COMPONENTS_GROUPING IGNORE)

#     set(CPACK_DEB_COMPONENT_INSTALL ON)
#     set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
#     set(CPACK_DEBIAN_ROCBLAS_RUNTIME_PACKAGE_NAME "rocblas")
#     set(CPACK_DEBIAN_ROCBLAS_RUNTIME_PACKAGE_SECTION "runtime")
#     set(CPACK_DEBIAN_ROCBLAS_DEVELOPMENT_PACKAGE_NAME "rocblas-dev")
#     set(CPACK_DEBIAN_ROCBLAS_DOCS_PACKAGE_NAME "rocblas-docs")
#     set(CPACK_DEBIAN_PACKAGE_PROVIDES "rocblas (= ${PROJECT_VERSION})")

#     set(CPACK_RPM_COMPONENT_INSTALL ON)
#     set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
#     set(CPACK_RPM_ROCBLAS_RUNTIME_PACKAGE_NAME "rocblas")
#     set(CPACK_RPM_ROCBLAS_DEVELOPMENT_PACKAGE_NAME "rocblas-devel")
#     set(CPACK_RPM_ROCBLAS_DOCS_PACKAGE_NAME "rocblas-docs")
#     set(CPACK_RPM_ROCBLAS_RUNTIME_PACKAGE_REQUIRES "rocblas (= ${PROJECT_VERSION})")
#     set(CPACK_RPM_PACKAGE_PROVIDES "rocblas (= ${PROJECT_VERSION})")

#     include(CPack)

#     cpack_add_component(rocBLAS_Runtime
#         DISPLAY_NAME "Runtime"
#         DESCRIPTION "Runtime libraries for rocBLAS"
#         REQUIRED
#     )
#     cpack_add_component(rocBLAS_Development
#         DISPLAY_NAME "Development"
#         DESCRIPTION "Headers and CMake files for rocBLAS"
#         DEPENDS rocBLAS_Runtime
#     )
#     cpack_add_component(rocBLAS_Docs
#         DISPLAY_NAME "Documentation"
#         DESCRIPTION "Documentation for rocBLAS"
#     )
# endif()
