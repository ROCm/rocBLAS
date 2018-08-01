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

/*! \brief BLAS EX API

    \details
    GEMM_EX performs one of the matrix-matrix operations

        D = alpha*op( A )*op( B ) + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, C, and D are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C and D m by n matrices.

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
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    Atype     rocblas_precision
              specifies the datatype of matrix A
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in]
    Btype     rocblas_precision
              specifies the datatype of matrix B
    @param[in]
    beta      specifies the scalar beta.
    @param[in]
    C         pointer storing matrix C on the GPU.
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.
    @param[in]
    Ctype     rocblas_precision
              specifies the datatype of matrix C
    @param[out]
    D         pointer storing matrix D on the GPU.
    @param[in]
    ldd       rocblas_int
              specifies the leading dimension of D.
    @param[in]
    Dtype     rocblas_precision
              specifies the datatype of matrix D

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

extern "C" rocblas_status rocblas_gemm_ex_template(
                           rocblas_handle        handle,
                           rocblas_operation     trans_a,
                           rocblas_operation     trans_b,
                           int                   m,
                           int                   n,
                           int                   k,
                           const float           *alpha,
                           const void            *a,
                           rocblas_precision     a_type,
                           int                   lda,
                           const void            *b,
                           rocblas_precision     b_type,
                           int                   ldb,
                           const float           *beta,
                           void                  *c,
                           rocblas_precision     c_type,
                           int                   ldc,
                           void                  *d,
                           rocblas_precision     d_type,
                           int                   ldd,
                           rocblas_precision     compute_type,
                           rocblas_gemm_algo     algo,
                           uint32_t              kernel_index,
                           uint32_t              flags)
{
    if(nullptr == handle)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_trace(handle,
                  "rocblas_gemm_ex",
                  trans_a, trans_b,
                  m, n, k, *alpha,
                  (const void*&)a, a_type, lda,
                  (const void*&)b, b_type, ldb, *beta,
                  (const void*&)c, c_type, ldc,
                  (const void*&)d, d_type, ldd,
                  compute_type,
                  algo,
                  kernel_index,
                  flags);        

        log_bench(handle,
                  "./rocblas-bench -f gemm-ex",
                  "-m", m,
                  "-n", n,
                  "-k", k,
                  "--alpha", *alpha,
                  "--beta", *beta,
                  "--lda", lda,
                  "--ldb", ldb,
                  "--ldc", ldc,
                  "--ldd", ldd,
                  "--a_type", a_type,
                  "--b_type", b_type,
                  "--c_type", c_type,
                  "--d_type", d_type,
                  "--compute_type", compute_type,
                  "--algo", algo,
                  "kernel_index", kernel_index,
                  "flags", flags);
    }
    else
    {
        log_trace(handle,
                  "rocblas_gemm_ex",
                  trans_a, trans_b,
                  m, n, k,
                  (const void*&)alpha,
                  (const void*&)a, a_type, lda, 
                  (const void*&)b, b_type, ldb, (const void*&)beta,
                  (const void*&)c, c_type, ldc,
                  (const void*&)d, d_type, ldd,
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
    if(a == nullptr || b == nullptr || c == nullptr || d == nullptr || alpha == nullptr || beta == nullptr)
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


//TODO: templated function needs to replace non-templated code below
//template <typename T>
//rocblas_matrix_copy_device_to_device(rocblas_int m, rocblas_int n, T *dest, rocblas_int ld_dest, T *orig, rocblas_int ld_orig)
//{
//    size_t byte_size = sizeof(T);
//    size_t column_byte_size = byte_size * m;
//    size_t dest_byte_stride = byte_size * ld_dest;
//    size_t orig_byte_stride = byte_size * ld_orig;
//
//    if((dest != orig) || (ld_dest != ld_orig))
//    {
//        for (int i = 0; i < n; i++)
//        {
//            void *c_void = static_cast<void*>(&(c_double[i*c_byte_stride]));
//            void *d_void = static_cast<void*>(&(d_double[i*d_byte_stride]));
//            PRINT_IF_HIP_ERROR(hipMemcpy(d_void, c_void, column_byte_size, hipMemcpyDeviceToDevice))
//        }
//    }
//} 




    if(a_type == rocblas_precision_double && b_type == rocblas_precision_double && 
       c_type == rocblas_precision_double && d_type == rocblas_precision_double && compute_type == rocblas_precision_double)
    {
        const double alpha_double = static_cast<double>(*alpha);
        const double beta_double = static_cast<double>(*beta);

        status = rocblas_dgemm(handle,
                               trans_a, trans_b,
                               m, n, k, static_cast<const double*>(&alpha_double),
                               static_cast<const double*>(a), lda,
                               static_cast<const double*>(b), ldb, static_cast<const double*>(&beta_double),
                               static_cast<      double*>(c), ldc);

        if(status != rocblas_status_success) return status;

        if(c != d || ldc != ldd)  // no copy if c matrix == d matrix
        {
            c_byte_size = sizeof(double);
            d_byte_size = sizeof(double);

            size_t column_size = m * c_byte_size;

            size_t c_byte_stride = ldc * c_byte_size;
            size_t d_byte_stride = ldd * d_byte_size;

            double *c_double = static_cast<double*>(c);
            double *d_double = static_cast<double*>(d);

            for (int i = 0; i < n; i++)
            {
                void *c_void = static_cast<void*>(&(c_double[i*c_byte_stride]));
                void *d_void = static_cast<void*>(&(d_double[i*d_byte_stride]));
                PRINT_IF_HIP_ERROR(hipMemcpy(d_void, c_void, column_size, hipMemcpyDeviceToDevice))
            }
        }

    }
    else if(a_type == rocblas_precision_single && b_type == rocblas_precision_single && 
            c_type == rocblas_precision_single && d_type == rocblas_precision_single && compute_type == rocblas_precision_single)
    {
        const float alpha_float = static_cast<float>(*alpha);
        const float beta_float = static_cast<float>(*beta);

        status = rocblas_sgemm(handle,
                               trans_a, trans_b,
                               m, n, k, static_cast<const float*>(&alpha_float),
                               static_cast<const float*>(a), lda,
                               static_cast<const float*>(b), ldb, static_cast<const float*>(&beta_float),
                               static_cast<      float*>(c), ldc);

        if(status != rocblas_status_success) return status;

        if(c != d || ldc != ldd)  // no copy if c matrix == d matrix
        {
            c_byte_size = sizeof(float);
            d_byte_size = sizeof(float);

            size_t column_size = m * c_byte_size;

            size_t c_byte_stride = ldc * c_byte_size;
            size_t d_byte_stride = ldd * d_byte_size;

            float *c_float = static_cast<float*>(c);
            float *d_float = static_cast<float*>(d);

            for (int i = 0; i < n; i++)
            {
                void *c_void = static_cast<void*>(&(c_float[i*c_byte_stride]));
                void *d_void = static_cast<void*>(&(d_float[i*d_byte_stride]));
                PRINT_IF_HIP_ERROR(hipMemcpy(d_void, c_void, column_size, hipMemcpyDeviceToDevice))
            }
        }
    }
//  else if(a_type == rocblas_precision_half && b_type == rocblas_precision_half && 
//          c_type == rocblas_precision_half && d_type == rocblas_precision_half && compute_type == rocblas_precision_half)
//  {
//      status = rocblas_hgemm(handle,
//                             trans_a, trans_b,
//                             m, n, k, alpha,
//                             a, lda,
//                             b, ldb, beta,
//                             c, ldc);
//      c_byte_size = 2;
//      d_byte_size = 2;
//  }
    else
    {
        return rocblas_status_not_implemented;
    }

/*
    //copy matrix c into matrix d
    if(status == rocblas_success)
    {
        size_t column_size = m * c_byte_size;
        size_t c_byte_stride = ldc * c_byte_size;
        size_t d_byte_stride = ldd * d_byte_size;
        if(c != d || ldc != ldd)  // no copy if c matrix == d matrix
        {
            for (int i = 0; i < n; i++)
            {
                PRINT_IF_HIP_ERROR(hipMemcpy(d + i*d_byte_stride, c + i*c_byte_stride, column_size, hipMemcpyDeviceToDevice))
            }
        }
    }
*/

    return status;
}
