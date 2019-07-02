# ########################################################################
# Copyright 2016-2019 Advanced Micro Devices, Inc.
# ########################################################################

# This file is intended to be used in two ways; independently in a stand alone PROJECT
# and as part of a superbuild.  If the file is included in a stand alone project, the
# variables are not expected to be preset, and this will produce options() in the GUI
# for the user to examine.  If this file is included in a superbuild, the options will be
# presented in the superbuild GUI, but then passed into the ExternalProject as -D
# parameters, which would already define them.

# Building tensile can add significant compile time; this option allows to build
# library without tensile to allow for rapid iteration without GEMM functionality
if( NOT BUILD_WITH_TENSILE )
  option( BUILD_WITH_TENSILE "Build rocBLAS with Tensile or not" ON )
endif( )

# Samples have no other dependencies except for rocblas, so are enabled by default
if( NOT BUILD_CLIENTS_SAMPLES )
  option( BUILD_CLIENTS_SAMPLES "Build rocBLAS samples" OFF )
endif( )

if( NOT BUILD_CLIENTS_TESTS )
  option( BUILD_CLIENTS_TESTS "Build rocBLAS unit tests" OFF )
endif( )

if( NOT BUILD_CLIENTS_BENCHMARKS )
  option( BUILD_CLIENTS_BENCHMARKS "Build rocBLAS benchmarks" OFF )
endif( )
