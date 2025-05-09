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
project(tensile VERSION 0.0.0 LANGUAGES CXX)

include(CMakeDependentOption)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TENSILE_ENABLE_HOST ON CACHE BOOL "Build device library.")
set(TENSILE_ENABLE_DEVICE ON CACHE BOOL "Build device libraries.")
set(TENSILE_ENABLE_CLIENT OFF CACHE BOOL "Build client app.")
set(TENSILE_BUILD_TESTING OFF CACHE BOOL "Build host library tests.")
set(TENSILE_ENABLE_HIP ON CACHE BOOL "")
set(TENSILE_BUILD_SHARED OFF CACHE BOOL "")
set(TENSILE_ENABLE_MSGPACK ON CACHE BOOL "")
cmake_dependent_option(TENSILE_ENABLE_LLVM "Use llvm yaml library" OFF "TENSILE_ENABLE_MSGPACK" ON)

set(TENSILE_ENABLE_OPENMP ON CACHE BOOL "Use OpenMP to improve performance.")
set(TENSILE_STATIC_ONLY ON CACHE BOOL "Disable exposing Tensile symbols in a shared library.")
set(GPU_TARGETS "" CACHE STRING "AMD gfx targets to crosscompile")

set(TENSILE_SOURCE_TREE "${CMAKE_CURRENT_SOURCE_DIR}/../Tensile/Source")

if(TENSILE_ENABLE_HOST)
    find_package(hip REQUIRED)
    if(TENSILE_BUILD_SHARED)
        add_library(tensile-host SHARED)
    else()
        add_library(tensile-host STATIC)
    endif()
    add_library(tensile::tensile-host ALIAS tensile-host)
    set_target_properties(tensile-host
        PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF)

    target_link_libraries(tensile-host
        PRIVATE
            hip::device
    )
    if(TENSILE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
        target_link_libraries(tensile-host PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(TENSILE_ENABLE_MSGPACK)
        find_package(msgpack-cxx CONFIG)
        if(msgpack-cxx_FOUND)
            message(STATUS "Found msgpack-cxx (>=6.x)")
        else()
            find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
            message(STATUS "Found msgpack (<=5.x)")
        endif()
        if(msgpack-cxx_FOUND)
            target_link_libraries(tensile-host PRIVATE msgpack-cxx)
        else()
            target_link_libraries(tensile-host PRIVATE msgpackc)
        endif()
        target_compile_definitions(tensile-host PUBLIC TENSILE_MSGPACK=1)
    endif()

    target_include_directories(tensile-host
        PUBLIC
            $<BUILD_INTERFACE:${TENSILE_SOURCE_TREE}/lib/include>
        PRIVATE
            "${TENSILE_SOURCE_TREE}/lib/include/Tensile"
    )

    if(TENSILE_ENABLE_LLVM)
        target_compile_definitions(tensile-host PUBLIC -DTENSILE_YAML=1)
        find_package(LLVM REQUIRED)
        find_library(LLVMObjectYAML_LIBRARY
            NAMES LLVMObjectYAML
            PATHS ${LLVM_LIBRARY_DIR}
        )
        if(LLVMObjectYAML_LIBRARY)
            target_link_libraries(tensile-host PRIVATE LLVMObjectYAML)
        else()
            target_link_libraries(tensile-host PRIVATE LLVM)
        endif()
        target_include_directories(tensile-host PRIVATE ${LLVM_INCLUDE_DIRS})
    endif()

    if(TENSILE_ENABLE_LLVM OR TENSILE_ENABLE_MSGPACK)
    target_compile_definitions(tensile-host PUBLIC TENSILE_DEFAULT_SERIALIZATION)
    endif()

    if(TENSILE_STATIC_ONLY)
        target_compile_definitions(tensile-host PUBLIC TENSILE_STATIC_ONLY)
    endif()

    target_compile_definitions(tensile-host PUBLIC TENSILE_USE_HIP)

    add_subdirectory(host)
endif()

if(TENSILE_ENABLE_DEVICE)
    add_subdirectory(device)
endif()

if(TENSILE_ENABLE_CLIENT)
    find_package(Boost REQUIRED COMPONENTS program_options)
    find_package(rocm_smi REQUIRED)
    add_library(tensile-client-helper)
    target_link_libraries(tensile-client-helper
        PRIVATE
            Boost::boost
            Boost::program_options
            hip::device
            rocm_smi64
            tensile-host
    )
    target_include_directories(tensile-client-helper
        PUBLIC
            $<BUILD_INTERFACE:${TENSILE_SOURCE_TREE}/client/include>
    )
    add_executable(tensile-client)
    target_link_libraries(tensile-client
        PRIVATE
            hip::device
            hip::host
            tensile-client-helper
            tensile-host
    )
    add_subdirectory(client)
endif()

if(TENSILE_BUILD_TESTING)
    find_package(Boost REQUIRED COMPONENTS filesystem)
    find_package(GTest REQUIRED)
    add_executable(tensile-tests)
    target_link_libraries(tensile-tests
        PRIVATE
            Boost::filesystem
            GTest::gtest
            tensile-host
            hip::host
            hip::device
    )

    target_include_directories(tensile-tests
        PRIVATE
            $<BUILD_INTERFACE:${TENSILE_SOURCE_TREE}/../../HostLibraryTests/testlib/include>
            $<BUILD_INTERFACE:${TENSILE_SOURCE_TREE}/client/include>
    )
    add_subdirectory(tests)
endif()
