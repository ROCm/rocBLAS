# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ###########################
# ROCm dependencies
# ###########################

include(FetchContent)

find_package(ROCM 0.11.0 CONFIG QUIET PATHS "${ROCM_PATH}") # First version with Sphinx doc gen improvement
if(NOT ROCM_FOUND)
  message(STATUS "ROCm CMake not found. Fetching...")
  set(rocm_cmake_tag
    "rocm-6.0.0"
    CACHE STRING "rocm-cmake tag to download")
  FetchContent_Declare(
    rocm-cmake
    GIT_REPOSITORY https://github.com/RadeonOpenCompute/rocm-cmake.git
    GIT_TAG        ${rocm_cmake_tag}
    SOURCE_SUBDIR "DISABLE ADDING TO BUILD" # We don't really want to consume the build and test targets of ROCm CMake.
  )
  FetchContent_MakeAvailable(rocm-cmake)
  find_package(ROCM CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
else()
  find_package(ROCM 0.11.0 CONFIG REQUIRED PATHS "${ROCM_PATH}")
endif()

# rocm-cmake helpers
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMHeaderWrapper)
include(ROCMClients)
