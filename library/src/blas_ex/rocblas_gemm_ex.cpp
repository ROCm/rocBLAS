/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "status.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

void device_matrix_copy(void* src,
                        rocblas_int ld_src,
                        void* dst,
                        rocblas_int ld_dst,
                        rocblas_int n1,
                        rocblas_int n2,
                        size_t elem_size)
{
    if((src != dst) || (ld_src != ld_dst)) // no copy if src matrix == dst matrix
    {
        if((n1 == ld_src) && (n1 == ld_dst))
        {
            // matrices C and D are contiguous, use single copy
            size_t matrix_size = n1 * n2 * elem_size;
            PRINT_IF_HIP_ERROR(hipMemcpy(dst, src, matrix_size, hipMemcpyDeviceToDevice))
        }
        else
        {
            size_t column_size = n1 * elem_size;

            for(int i2 = 0; i2 < n2; i2++)
            {
                void* src_void =
                    static_cast<void*>(static_cast<uint8_t*>(src) + (i2 * ld_src * elem_size));
                void* dst_void =
                    static_cast<void*>(static_cast<uint8_t*>(dst) + (i2 * ld_dst * elem_size));

                PRINT_IF_HIP_ERROR(
                    hipMemcpy(dst_void, src_void, column_size, hipMemcpyDeviceToDevice))
            }
        }
    }
}

/*! \brief BLAS EX API

    \details
    GEMM_EX performs one of the matrix-matrix operations

        D = alpha*op( A )*op( B ) + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, C, and D are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C and D are m by n matrices.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    transA    rocblas_operation
              specifies the form of op( A )
    @param[in]
    transB    rocblas_operation
              specifies the form of op( B )
    @param[in]
    m         rocblas_int.
    @param[in]
    n         rocblas_int.
    @param[in]
    k         rocblas_int.
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    Atype     rocblas_precision
              specifies the datatype of matrix A
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    Btype     rocblas_precision
              specifies the datatype of matrix B
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in]
    beta      specifies the scalar beta.
    @param[in]
    C         pointer storing matrix C on the GPU.
    @param[in]
    Ctype     rocblas_precision
              specifies the datatype of matrix C
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.
    @param[out]
    D         pointer storing matrix D on the GPU.
    @param[in]
    @param[in]
    Dtype     rocblas_precision
              specifies the datatype of matrix D
    ldd       rocblas_int
              specifies the leading dimension of D.
    @param[in]
    computeType
              specifies the datatype of computation
    @param[in]
    algo      rocblas_gemm_algo
              enumerant specifying the algorithm type.
    @param[in]
    kernel_index
              reserved for future use
    @param[in]
    flags     uint32_t
              reserved for future use


    ********************************************************************/

extern "C" rocblas_status rocblas_gemm_ex(rocblas_handle handle,
                                          rocblas_operation trans_a,
                                          rocblas_operation trans_b,
                                          int m,
                                          int n,
                                          int k,
                                          const float* alpha,
                                          const void* a,
                                          rocblas_precision a_type,
                                          int lda,
                                          const void* b,
                                          rocblas_precision b_type,
                                          int ldb,
                                          const float* beta,
                                          void* c,
                                          rocblas_precision c_type,
                                          int ldc,
                                          void* d,
                                          rocblas_precision d_type,
                                          int ldd,
                                          rocblas_precision compute_type,
                                          rocblas_gemm_algo algo,
                                          uint32_t kernel_index,
                                          uint32_t flags)
{
    if(nullptr == handle)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_trace(handle,
                  "rocblas_gemm_ex",
                  trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  *alpha,
                  (const void*&)a,
                  a_type,
                  lda,
                  (const void*&)b,
                  b_type,
                  ldb,
                  *beta,
                  (const void*&)c,
                  c_type,
                  ldc,
                  (const void*&)d,
                  d_type,
                  ldd,
                  compute_type,
                  algo,
                  kernel_index,
                  flags);

        std::string trans_a_letter = rocblas_transpose_letter(trans_a);
        std::string trans_b_letter = rocblas_transpose_letter(trans_b);
        log_bench(handle,
                  "./rocblas-bench -f gemm_ex",
                  "--transposeA",
                  trans_a_letter,
                  "--transposeB",
                  trans_b_letter,
                  "-m",
                  m,
                  "-n",
                  n,
                  "-k",
                  k,
                  "--alpha",
                  *alpha,
                  "--a_type",
                  a_type,
                  "--lda",
                  lda,
                  "--b_type",
                  b_type,
                  "--ldb",
                  ldb,
                  "--beta",
                  *beta,
                  "--c_type",
                  c_type,
                  "--ldc",
                  ldc,
                  "--d_type",
                  d_type,
                  "--ldd",
                  ldd,
                  "--compute_type",
                  compute_type,
                  "--algo",
                  algo,
                  "--kernel_index",
                  kernel_index,
                  "--flags",
                  flags);
    }
    else
    {
        log_trace(handle,
                  "rocblas_gemm_ex",
                  trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  (const void*&)alpha,
                  (const void*&)a,
                  a_type,
                  lda,
                  (const void*&)b,
                  b_type,
                  ldb,
                  (const void*&)beta,
                  (const void*&)c,
                  c_type,
                  ldc,
                  (const void*&)d,
                  d_type,
                  ldd,
                  compute_type,
                  algo,
                  kernel_index,
                  flags);
    }

    // quick return m,n,k equal to 0 is valid in BLAS
    if(m == 0 || n == 0 || k == 0)
    {
        return rocblas_status_success;
    }

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0)
    {
        return rocblas_status_invalid_size;
    }

    // pointers must be valid
    if(a == nullptr || b == nullptr || c == nullptr || d == nullptr || alpha == nullptr ||
       beta == nullptr)
    {
        return rocblas_status_invalid_pointer;
    }

    rocblas_int num_rows_a = (trans_a == rocblas_operation_none) ? m : k;
    rocblas_int num_rows_b = (trans_b == rocblas_operation_none) ? k : n;
    rocblas_int num_rows_c = m;
    rocblas_int num_rows_d = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc || num_rows_d > ldd)
    {
        return rocblas_status_invalid_size;
    }

    rocblas_status status;
    size_t c_byte_size;
    size_t d_byte_size;

    if(a_type == rocblas_precision_double && b_type == rocblas_precision_double &&
       c_type == rocblas_precision_double && d_type == rocblas_precision_double &&
       compute_type == rocblas_precision_double)
    {
        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            // copy alpha and beta from device to host to convert type, then copy back to device
            float h_alpha_float;
            float h_beta_float;
            hipMemcpy(&h_alpha_float, alpha, sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(&h_beta_float, beta, sizeof(float), hipMemcpyDeviceToHost);

            const double h_alpha_double = static_cast<double>(h_alpha_float);
            const double h_beta_double  = static_cast<double>(h_beta_float);

            double* d_alpha_double;
            double* d_beta_double;

            hipMalloc(&d_alpha_double, sizeof(double));
            hipMalloc(&d_beta_double, sizeof(double));

            hipMemcpy(d_alpha_double, &h_alpha_double, sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(d_beta_double, &h_beta_double, sizeof(double), hipMemcpyHostToDevice);

            // copy matrix C to matrix D
            device_matrix_copy(c, ldc, d, ldd, m, n, sizeof(double));

            // call rocblas_dgemm
            status = rocblas_dgemm(handle,
                                   trans_a,
                                   trans_b,
                                   m,
                                   n,
                                   k,
                                   d_alpha_double,
                                   static_cast<const double*>(a),
                                   lda,
                                   static_cast<const double*>(b),
                                   ldb,
                                   d_beta_double,
                                   static_cast<double*>(d),
                                   ldd);
        }
        else
        {
            // convert type of alpha and beta
            const double alpha_double = static_cast<double>(*alpha);
            const double beta_double  = static_cast<double>(*beta);

            // copy matrix C to matrix D
            device_matrix_copy(c, ldc, d, ldd, m, n, sizeof(double));

            // call rocblas_dgemm
            status = rocblas_dgemm(handle,
                                   trans_a,
                                   trans_b,
                                   m,
                                   n,
                                   k,
                                   static_cast<const double*>(&alpha_double),
                                   static_cast<const double*>(a),
                                   lda,
                                   static_cast<const double*>(b),
                                   ldb,
                                   static_cast<const double*>(&beta_double),
                                   static_cast<double*>(d),
                                   ldd);
        }

        if(status != rocblas_status_success)
            return status;
    }
    else if(a_type == rocblas_precision_single && b_type == rocblas_precision_single &&
            c_type == rocblas_precision_single && d_type == rocblas_precision_single &&
            compute_type == rocblas_precision_single)
    {

        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            // no need for type conversion for alpha and beta

            // copy matrix C to matrix D
            device_matrix_copy(c, ldc, d, ldd, m, n, sizeof(float));

            // call rocblas_sgemm
            status = rocblas_sgemm(handle,
                                   trans_a,
                                   trans_b,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   static_cast<const float*>(a),
                                   lda,
                                   static_cast<const float*>(b),
                                   ldb,
                                   beta,
                                   static_cast<float*>(d),
                                   ldd);
        }
        else
        {
            // no need for type conversion for alpha and beta

            // copy matrix C to matrix D
            device_matrix_copy(c, ldc, d, ldd, m, n, sizeof(float));

            // call rocblas_sgemm
            status = rocblas_sgemm(handle,
                                   trans_a,
                                   trans_b,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   static_cast<const float*>(a),
                                   lda,
                                   static_cast<const float*>(b),
                                   ldb,
                                   beta,
                                   static_cast<float*>(d),
                                   ldd);
        }

        if(status != rocblas_status_success)
            return status;
    }
    else if(a_type == rocblas_precision_half && b_type == rocblas_precision_half &&
            c_type == rocblas_precision_half && d_type == rocblas_precision_half &&
            compute_type == rocblas_precision_half)
    {
        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            // copy alpha and beta from device to host to convert type, then copy back to device
            float h_alpha_float;
            float h_beta_float;
            hipMemcpy(&h_alpha_float, alpha, sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(&h_beta_float, beta, sizeof(float), hipMemcpyDeviceToHost);

            const _Float16 h_alpha_half = static_cast<_Float16>(h_alpha_float);
            const _Float16 h_beta_half  = static_cast<_Float16>(h_beta_float);

            rocblas_half* d_alpha_half;
            rocblas_half* d_beta_half;

            hipMalloc(&d_alpha_half, sizeof(rocblas_half));
            hipMalloc(&d_beta_half, sizeof(rocblas_half));

            hipMemcpy(d_alpha_half, &h_alpha_half, sizeof(rocblas_half), hipMemcpyHostToDevice);
            hipMemcpy(d_beta_half, &h_beta_half, sizeof(rocblas_half), hipMemcpyHostToDevice);

            // copy matrix C to matrix D
            device_matrix_copy(c, ldc, d, ldd, m, n, sizeof(rocblas_half));

            // call rocblas_hgemm
            status = rocblas_hgemm(handle,
                                   trans_a,
                                   trans_b,
                                   m,
                                   n,
                                   k,
                                   d_alpha_half,
                                   static_cast<const rocblas_half*>(a),
                                   lda,
                                   static_cast<const rocblas_half*>(b),
                                   ldb,
                                   d_beta_half,
                                   static_cast<rocblas_half*>(d),
                                   ldd);
        }
        else
        {
            // convert type of alpha and beta
            const _Float16 alpha_half = static_cast<_Float16>(*alpha);
            const _Float16 beta_half  = static_cast<_Float16>(*beta);

            // copy matrix C to matrix D
            device_matrix_copy(c, ldc, d, ldd, m, n, sizeof(rocblas_half));

            // call rocblas_hgemm
            status = rocblas_hgemm(handle,
                                   trans_a,
                                   trans_b,
                                   m,
                                   n,
                                   k,
                                   reinterpret_cast<const rocblas_half*>(&alpha_half),
                                   static_cast<const rocblas_half*>(a),
                                   lda,
                                   static_cast<const rocblas_half*>(b),
                                   ldb,
                                   reinterpret_cast<const rocblas_half*>(&beta_half),
                                   static_cast<rocblas_half*>(d),
                                   ldd);
        }

        if(status != rocblas_status_success)
            std::cout << "ERROR: status = " << status << std::endl;
        if(status != rocblas_status_success)
            return status;
    }
    else
    {
        return rocblas_status_not_implemented;
    }

    return status;
}
