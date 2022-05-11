# ########################################################################
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

# CMake v3.21.0 can identify ROCMClang, but there are some issues with feature detection.
# For now, disable ROCMClang detection to make it work the same as CMake v3.20 and earlier.
set(__skip_rocmclang ON)


# This will add compile option: -std=c++17
set( CMAKE_CXX_STANDARD 17 )
# Without this line, it will add -std=gnu++17 instead, which may have issues.
set( CMAKE_CXX_EXTENSIONS OFF )

# disable this next option and test build whenever compiler changes to verify unused arguments are safe to ignore
# add_compile_options( -Wno-unused-command-line-argument )

# TODO: address [[nodiscard]] warnings
add_compile_options( $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result> )


# ########################################################################
# NOTE:  CUDA compiling path
# ########################################################################
# I have tried compiling rocBLAS library source with multiple methods,
# and ended up using the approach where we set the CXX compiler to hipcc.
# I didn't like using the HIP_ADD_LIBRARY or CUDA_ADD_LIBRARY approaches,
# for the reasons I list here.
# 1.  Adding header include directories is through HIP_INCLUDE_DIRECTORIES(), which
# is global to a directory and affects all targets
# 2.  You must add HIP_SOURCE_PROPERTY_FORMAT OBJ properties to .cpp files
# to get HIP_ADD_LIBRARY to recognize the file
# 3.  HIP_ADD_LIBRARY invokes a call to add_custom_command() to compile files,
# and rocBLAS does the same.  The order in which custom commands execute is
# undefined, and sometimes a file is attempted to be compiled before it has
# been generated.  The fix for this is to create 'PHONY' targets, which I
# don't desire.

# Using hipcc allows us to avoid the above problems, with two primary costs:
# 1.  The cmake logic to detect compiler features fails with nvcc backend
# 2.  Upfront cost to figure out all the strange compiler/linker flags I define
# below.

# Hopefully, cost #2 is already paid.  All in all, I want to get rid of the
# need for hipcc, and hope that at some point of time in the future we
# can use the export config files from hip for both ROCm & nvcc backends.
# ########################################################################

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  # Determine if CXX Compiler is hip-clang or nvcc
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
  string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})
endif()

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message( STATUS "Use hip-clang to build for amdgpu backend" )

  # set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fallow-half-arguments-and-returns" )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1" )

  if ( CMAKE_BUILD_TYPE MATCHES "Debug" AND NOT WIN32 )
    set ( CMAKE_CXX_FLAGS_DEBUG "-O1 ${CMAKE_CXX_FLAGS_DEBUG} -gsplit-dwarf -ggdb" )
  endif()

elseif( CXX_VERSION_STRING MATCHES "nvcc" )
  message( STATUS "HIPCC nvcc compiler detected; CUDA backend selected" )

  set( CMAKE_C_COMPILE_OPTIONS_PIC "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_PIC}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_PIC "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )
  set( CMAKE_SHARED_LIBRARY_C_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
  set( CMAKE_SHARED_LIBRARY_CXX_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
  set( CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
  set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_VISIBILITY}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY}" )
  set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}" )
elseif( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  message( STATUS "ERROR: HCC compiler is no longer supported!" )
endif( )


# FOR OPTIONAL CODE COVERAGE
if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

if(BUILD_ADDRESS_SANITIZER AND BUILD_SHARED_LIBS)
  # Fortran not supported, add_link_options below invalid for fortran linking
  set(BUILD_FORTRAN_CLIENTS OFF)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()



