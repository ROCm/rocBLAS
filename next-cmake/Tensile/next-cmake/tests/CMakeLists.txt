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

if(TENSILE_ENABLE_LLVM)
    add_subdirectory(llvm)
endif()
if(TENSILE_ENABLE_HIP)
    add_subdirectory(hip)
endif()

set(_CMAKE_CURRENT_SOURCE_DIR "${TENSILE_SOURCE_TREE}/../../HostLibraryTests")

target_sources(tensile-tests
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/ContractionFitness_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/EmbeddedData_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DecisionTree_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ContractionProblem_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ContractionSelectionLibrary_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ContractionLibraryLoading_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Utils_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ProjectedPerformance_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DataTypes_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MultipleSolutionsPerSize_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TestData_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/testlib/source/TestData.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CachingLibrary_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelArguments_test.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Predicates_test.cpp"
)

add_subdirectory("${_CMAKE_CURRENT_SOURCE_DIR}/configs" "${CMAKE_CURRENT_BINARY_TREE}/configs")

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS ${FILE})
        message(WARNING "Overwriting ${FILE} from ${FILE}.gz")
    endif()
    execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach(FILE)
