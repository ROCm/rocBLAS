# ########################################################################
# Copyright 2016-2022 Advanced Micro Devices, Inc.
# ########################################################################

# This file is intended to be used in two ways; independently in a stand alone PROJECT
# and as part of a superbuild.  If the file is included in a stand alone project, the
# variables are not expected to be preset, and this will produce options() in the GUI
# for the user to examine.  If this file is included in a superbuild, the options will be
# presented in the superbuild GUI, but then passed into the ExternalProject as -D
# parameters, which would already define them.

option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocBLAS as a shared library" ON )

option( BUILD_TESTING "Build tests for rocBLAS" ON )


# Building tensile can add significant compile time; this option allows to build
# library without tensile to allow for rapid iteration without GEMM functionality
option( BUILD_WITH_TENSILE "Build full functionality which requires tensile?" ON )

include(clients/cmake/client-build-options.cmake)

if (WIN32)
  # not supported on windows so set off
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

# this file is intended to be loaded by toolchain or early as sets global compiler flags
# rocm-cmake checks will throw warnings if set later as cmake watchers installed

# FOR OPTIONAL CODE COVERAGE
option(BUILD_CODE_COVERAGE "Build rocBLAS with code coverage enabled" OFF)

# FOR OPTIONAL ADDRESS SANITIZER
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)

if( BUILD_WITH_TENSILE )

  set( Tensile_CPU_THREADS "" CACHE STRING "Number of threads for Tensile parallel build")

  set( Tensile_LOGIC "asm_full" CACHE STRING "Tensile to use which logic?")
  set( Tensile_CODE_OBJECT_VERSION "V2" CACHE STRING "Tensile code_object_version")
  set( Tensile_COMPILER "hipcc" CACHE STRING "Tensile compiler")
  set( Tensile_LIBRARY_FORMAT "msgpack" CACHE STRING "Tensile library format")

  set_property( CACHE Tensile_LOGIC PROPERTY STRINGS aldebaran asm_full asm_lite asm_miopen hip_lite other )
  set_property( CACHE Tensile_CODE_OBJECT_VERSION PROPERTY STRINGS V2 V3 )
  set_property( CACHE Tensile_COMPILER PROPERTY STRINGS hcc hipcc)
  set_property( CACHE Tensile_LIBRARY_FORMAT PROPERTY STRINGS msgpack yaml)

  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
  option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )

  if(Tensile_LIBRARY_FORMAT MATCHES "yaml")
    option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
    option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." OFF)
  else()
    option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." OFF)
    option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." ON)
  endif()

  option( TENSILE_VENV_UPGRADE_PIP "Upgrade pip in Tensile virtuaal environment" OFF)

endif()


# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)
