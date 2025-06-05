# ##############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#
# ##############################################################################

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/gtest)

add_executable(rocblas-test)

set_target_properties(rocblas-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              "${PROJECT_BINARY_DIR}/staging")

target_compile_definitions(rocblas-test PRIVATE GOOGLE_TEST)
if(ROCBLAS_ENABLE_TENSILE)
  target_compile_definitions(rocblas-test PRIVATE BUILD_WITH_TENSILE)
endif()

target_link_libraries(rocblas-test PRIVATE rocblas-clients-common)

add_subdirectory(include)
add_subdirectory(src)

set(yaml_test_files
    "${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/asum_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/atomics_mode_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/aux_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/axpy_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/blas1_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/blas1_gtest.yaml.new"
    "${_CMAKE_CURRENT_SOURCE_DIR}/blas2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/blas3_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/copy_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/dgmm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/dot_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gbmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/geam_ex_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/geam_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemm_batched_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemm_strided_batched_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemmt_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/general_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/ger_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/geruc_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/get_solutions_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hbmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hemm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hemv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/her2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/her2k_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/her_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/herk_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/herkx_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hpmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hpr2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hpr_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/iamax_iamin_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/known_bugs.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/logging_mode_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/multiheaded_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/nrm2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/ostream_threadsafety_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/rot_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/sbmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/scal_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/set_get_atomics_mode_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/set_get_matrix_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/set_get_pointer_mode_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/set_get_vector_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/spmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/spr2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/spr_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/swap_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/symm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/symv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syr2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syr2k_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syr_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syrk_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syrkx_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/tbmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/tbsv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/tpmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/tpsv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trmm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trsm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trsv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trtri_gtest.yaml")

set(ROCBLAS_TEST_DATA
    "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_gtest.data")

add_custom_command(
  OUTPUT "${ROCBLAS_TEST_DATA}"
  COMMAND
    "${python}" ARGS "${_CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_gentest.py"
    -I "${_CMAKE_CURRENT_SOURCE_DIR}/../include" "rocblas_gtest.yaml" -o
    "${ROCBLAS_TEST_DATA}"
  DEPENDS "${_CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_gentest.py"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_common.yaml"
          ${yaml_test_files}
  WORKING_DIRECTORY "${_CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND_EXPAND_LISTS VERBATIM)

add_custom_target(
  rocblas-test-data ALL
  DEPENDS "${ROCBLAS_TEST_DATA}"
  COMMENT "Generating rocBLAS test data")

set_target_properties(rocblas-test-data PROPERTIES ROCBLAS_INSTALL_FILES "${ROCBLAS_TEST_DATA}")

function(rocblas_add_copy_command OUTPUT_FILE SOURCE_FILE COMMENT_MSG)
  add_custom_command(
    OUTPUT "${OUTPUT_FILE}"
    COMMAND "${CMAKE_COMMAND}" ARGS -E copy "${SOURCE_FILE}" "${OUTPUT_FILE}"
    DEPENDS "${SOURCE_FILE}"
    WORKING_DIRECTORY "${_CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "${COMMENT_MSG}"
    VERBATIM)
endfunction()

set(ROCBLAS_SMOKE "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_smoke.yaml")
rocblas_add_copy_command(
  "${ROCBLAS_SMOKE}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_smoke.yaml"
  "Copying rocBLAS smoke test configuration")

set(ROCBLAS_EXTRAS "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_extras.yaml")
rocblas_add_copy_command(
  "${ROCBLAS_EXTRAS}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_extras.yaml"
  "Copying rocBLAS extras configuration")

set(ROCBLAS_CLIENTS_RTEST
    "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_rtest.py")
rocblas_add_copy_command(
  "${ROCBLAS_CLIENTS_RTEST}" "${_CMAKE_CURRENT_SOURCE_DIR}/../../rtest.py"
  "Copying rocBLAS test scripts")

set(ROCBLAS_CLIENTS_RTEST_XML
    "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_rtest.xml")
rocblas_add_copy_command(
  "${ROCBLAS_CLIENTS_RTEST_XML}" "${_CMAKE_CURRENT_SOURCE_DIR}/../../rtest.xml"
  "Copying rocBLAS test XML")

set(ROCBLAS_COMMON "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_common.yaml")
rocblas_add_copy_command(
  "${ROCBLAS_COMMON}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_common.yaml"
  "Copying rocBLAS common configuration")

set(ROCBLAS_TEMPLATE
    "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_template.yaml")
rocblas_add_copy_command(
  "${ROCBLAS_TEMPLATE}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_template.yaml"
  "Copying rocBLAS template configuration")

set(ROCBLAS_GENERAL_YAML
    "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_general.yaml")
rocblas_add_copy_command(
  "${ROCBLAS_GENERAL_YAML}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_general.yaml"
  "Copying rocBLAS general configuration")

set(ROCBLAS_GENTEST "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_gentest.py")
rocblas_add_copy_command(
  "${ROCBLAS_GENTEST}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_gentest.py"
  "Copying rocBLAS gentest script")

set(ROCBLAS_CLIENTS_README
    "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_clients_readme.txt")
rocblas_add_copy_command(
  "${ROCBLAS_CLIENTS_README}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../rocblas_clients_readme.txt"
  "Copying rocBLAS clients readme")

# I think we should move this up into the clients/CMakeLists.txt file
add_custom_target(
  rocblas-test-files ALL
  DEPENDS "${ROCBLAS_SMOKE}"
          "${ROCBLAS_EXTRAS}"
          "${ROCBLAS_CLIENTS_RTEST}"
          "${ROCBLAS_CLIENTS_RTEST_XML}"
          "${ROCBLAS_COMMON}"
          "${ROCBLAS_TEMPLATE}"
          "${ROCBLAS_GENERAL_YAML}"
          "${ROCBLAS_GENTEST}"
          "${ROCBLAS_CLIENTS_README}"
  COMMENT "Generating rocBLAS test files")

set_target_properties(
  rocblas-test-files
  PROPERTIES
    ROCBLAS_INSTALL_FILES
    "${ROCBLAS_SMOKE};${ROCBLAS_EXTRAS};${ROCBLAS_COMMON};${ROCBLAS_TEMPLATE};${ROCBLAS_GENERAL_YAML};${ROCBLAS_CLIENTS_README};${ROCBLAS_CLIENTS_RTEST_XML}"
    ROCBLAS_INSTALL_PROGRAMS "${ROCBLAS_CLIENTS_RTEST};${ROCBLAS_GENTEST}")

add_dependencies(rocblas-test rocblas-test-data rocblas-test-files)
