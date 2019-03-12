#!/bin/bash
bench=${bench}
if [ ! -f ${bench} ]; then
	echo ${bench} not found, exit...
	exit 1
else
	echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi

echo "-----------------------------------------------------------------------------------------------------"
echo "----------i8_r-i32_r-gemm_ex-------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------------------------------"
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3025 -n 256 -k 64  --lda 3025 --ldb 64  --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3025 -n 64  -k 64  --lda 3025 --ldb 64  --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3025 -n 64  -k 256 --lda 3025 --ldb 256 --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 784  -n 512 -k 128 --lda 784  --ldb 128 --ldc 784  --ldd 784  --batch 64
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 784  -n 128 -k 512 --lda 784  --ldb 512 --ldc 784  --ldd 784  --batch 64

echo "-----------------------------------------------------------------------------------------------------"
echo "----------f32_r-gemm_ex------------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------------------------------"
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3025 -n 256 -k 16  --lda 3025 --ldb 16  --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3025 -n 64  -k 16  --lda 3025 --ldb 16  --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3025 -n 64  -k 64  --lda 3025 --ldb 64  --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 784  -n 512 -k 32  --lda 784  --ldb 32  --ldc 784  --ldd 784  --batch 64
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 784  -n 128 -k 128 --lda 784  --ldb 128 --ldc 784  --ldd 784  --batch 64

