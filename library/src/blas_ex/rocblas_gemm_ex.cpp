/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "Tensile.h"
#include "TensileTypes.h"
#include "status.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"
#include <type_traits>
#include "rocblas_gemm_ex.hpp"

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
              matrix dimension m
    @param[in]
    n         rocblas_int.
              matrix dimension n
    @param[in]
    k         rocblas_int.
              matrix dimension k
    @param[in]
    alpha     const void *
              specifies the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         void *
              pointer storing matrix A on the GPU.
    @param[in]
    a_type    rocblas_datatype
              specifies the datatype of matrix A
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    b         void *
              pointer storing matrix B on the GPU.
    @param[in]
    b_type    rocblas_datatype
              specifies the datatype of matrix B
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in]
    beta      const void *
              specifies the scalar beta. Same datatype as compute_type.
    @param[in]
    c         void *
              pointer storing matrix C on the GPU.
    @param[in]
    c_type    rocblas_datatype
              specifies the datatype of matrix C
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.
    @param[out]
    d         void *
              pointer storing matrix D on the GPU.
    @param[in]
    d_type    rocblas_datatype
              specifies the datatype of matrix D
    @param[in]
    ldd       rocblas_int
              specifies the leading dimension of D.
    @param[in]
    compute_type
              rocblas_datatype
              specifies the datatype of computation
    @param[in]
    algo      rocblas_gemm_algo
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              int32_t
              reserved for future use
    @param[in]
    flags     uint32_t
              reserved for future use
    @param[in/out]
    workspace_size
              size_t*
              size of workspace
    @parm[in]
    workspace void*
              workspace

    ********************************************************************/

extern "C" rocblas_status rocblas_gemm_ex(rocblas_handle handle,
                                          rocblas_operation trans_a,
                                          rocblas_operation trans_b,
                                          rocblas_int m,
                                          rocblas_int n,
                                          rocblas_int k,
                                          const void* alpha,
                                          const void* a,
                                          rocblas_datatype a_type,
                                          rocblas_int lda,
                                          const void* b,
                                          rocblas_datatype b_type,
                                          rocblas_int ldb,
                                          const void* beta,
                                          const void* c,
                                          rocblas_datatype c_type,
                                          rocblas_int ldc,
                                          void* d,
                                          rocblas_datatype d_type,
                                          rocblas_int ldd,
                                          rocblas_datatype compute_type,
                                          rocblas_gemm_algo algo,
                                          int32_t solution_index,
                                          uint32_t flags,
                                          size_t* workspace_size,
                                          void* workspace)
{
    // handle, alpha, beta must not be null pointers for logging
    if(!handle)
        return rocblas_status_invalid_handle;

    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        char trans_a_letter, trans_b_letter;
        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            trans_a_letter = rocblas_transpose_letter(trans_a);
            trans_b_letter = rocblas_transpose_letter(trans_b);
        }
        auto a_type_string       = rocblas_datatype_string(a_type);
        auto b_type_string       = rocblas_datatype_string(b_type);
        auto c_type_string       = rocblas_datatype_string(c_type);
        auto d_type_string       = rocblas_datatype_string(d_type);
        auto compute_type_string = rocblas_datatype_string(compute_type);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_trace))
        {
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                double alpha_double;
                double beta_double;
                if(compute_type == rocblas_datatype_f16_r)
                {
                    alpha_double = *static_cast<const _Float16*>(alpha);
                    beta_double  = *static_cast<const _Float16*>(beta);
                }
                else if(compute_type == rocblas_datatype_f32_r)
                {
                    alpha_double = *static_cast<const float*>(alpha);
                    beta_double  = *static_cast<const float*>(beta);
                }
                else if(compute_type == rocblas_datatype_f64_r)
                {
                    alpha_double = *static_cast<const double*>(alpha);
                    beta_double  = *static_cast<const double*>(beta);
                }
                else if(compute_type == rocblas_datatype_i32_r)
                {
                    alpha_double = *static_cast<const int32_t*>(alpha);
                    beta_double  = *static_cast<const int32_t*>(beta);
                }

                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              "rocblas_gemm_ex",
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              alpha_double,
                              a,
                              a_type_string,
                              lda,
                              b,
                              b_type_string,
                              ldb,
                              beta_double,
                              c,
                              c_type_string,
                              ldc,
                              d,
                              d_type_string,
                              ldd,
                              compute_type_string,
                              algo,
                              solution_index,
                              flags,
                              workspace_size ? *workspace_size : 0,
                              workspace);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
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
                              alpha_double,
                              "--a_type",
                              a_type_string,
                              "--lda",
                              lda,
                              "--b_type",
                              b_type_string,
                              "--ldb",
                              ldb,
                              "--beta",
                              beta_double,
                              "--c_type",
                              c_type_string,
                              "--ldc",
                              ldc,
                              "--d_type",
                              d_type_string,
                              "--ldd",
                              ldd,
                              "--compute_type",
                              compute_type_string,
                              "--algo",
                              algo,
                              "--solution_index",
                              solution_index,
                              "--flags",
                              flags,
                              "--workspace_size",
                              workspace_size ? *workspace_size : 0);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              "rocblas_gemm_ex",
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              alpha,
                              a,
                              a_type_string,
                              lda,
                              b,
                              b_type_string,
                              ldb,
                              beta,
                              c,
                              c_type_string,
                              ldc,
                              d,
                              d_type_string,
                              ldd,
                              compute_type_string,
                              algo,
                              solution_index,
                              flags,
                              "--workspace_size",
                              workspace_size ? *workspace_size : 0);
            }
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            log_profile(handle,
                        "rocblas_gemm_ex",
                        "a_type",
                        a_type_string,
                        "b_type",
                        b_type_string,
                        "c_type",
                        c_type_string,
                        "d_type",
                        d_type_string,
                        "compute_type",
                        compute_type_string,
                        "transA",
                        trans_a_letter,
                        "transB",
                        trans_b_letter,
                        "M",
                        m,
                        "N",
                        n,
                        "K",
                        k,
                        "lda",
                        lda,
                        "ldb",
                        ldb,
                        "ldc",
                        ldc,
                        "ldd",
                        ldd,
                        "algo",
                        algo,
                        "solution_index",
                        solution_index,
                        "flags",
                        flags,
                        "workspace_size",
                        workspace_size ? *workspace_size : 0);
        }
    }

    // quick return m,n,k equal to 0 is valid in BLAS
    if(!m || !n || !k)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0)
        return rocblas_status_invalid_size;

    // pointers must be valid
    if(!a || !b || !c || !d)
        return rocblas_status_invalid_pointer;

    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;
    rocblas_int num_rows_c = m;
    rocblas_int num_rows_d = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc || num_rows_d > ldd)
        return rocblas_status_invalid_size;

    rocblas_status rb_status = rocblas_status_internal_error;
    rocblas_int batch_count  = 1;
    rocblas_int stride_a     = trans_a == rocblas_operation_none ? lda * k : lda * m;
    rocblas_int stride_b     = trans_b == rocblas_operation_none ? ldb * n : ldb * k;
    rocblas_int stride_c     = ldc * n;
    rocblas_int stride_d     = ldd * n;

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r &&
       c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r &&
       compute_type == rocblas_datatype_f64_r)
    {
        rb_status = gemm_ex_typecasting<double, double, double>(handle,
                                                                trans_a,
                                                                trans_b,
                                                                m,
                                                                n,
                                                                k,
                                                                alpha,
                                                                a,
                                                                lda,
                                                                stride_a,
                                                                b,
                                                                ldb,
                                                                stride_b,
                                                                beta,
                                                                c,
                                                                ldc,
                                                                stride_c,
                                                                d,
                                                                ldd,
                                                                stride_d,
                                                                batch_count);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r &&
            c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r &&
            compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<float, float, float>(handle,
                                                             trans_a,
                                                             trans_b,
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             a,
                                                             lda,
                                                             stride_a,
                                                             b,
                                                             ldb,
                                                             stride_b,
                                                             beta,
                                                             c,
                                                             ldc,
                                                             stride_c,
                                                             d,
                                                             ldd,
                                                             stride_d,
                                                             batch_count);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
            c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r &&
            compute_type == rocblas_datatype_f16_r)
    {
        rb_status = gemm_ex_typecasting<_Float16, _Float16, _Float16>(handle,
                                                                      trans_a,
                                                                      trans_b,
                                                                      m,
                                                                      n,
                                                                      k,
                                                                      alpha,
                                                                      a,
                                                                      lda,
                                                                      stride_a,
                                                                      b,
                                                                      ldb,
                                                                      stride_b,
                                                                      beta,
                                                                      c,
                                                                      ldc,
                                                                      stride_c,
                                                                      d,
                                                                      ldd,
                                                                      stride_d,
                                                                      batch_count);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
            c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r &&
            compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<_Float16, _Float16, float>(handle,
                                                                   trans_a,
                                                                   trans_b,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   alpha,
                                                                   a,
                                                                   lda,
                                                                   stride_a,
                                                                   b,
                                                                   ldb,
                                                                   stride_b,
                                                                   beta,
                                                                   c,
                                                                   ldc,
                                                                   stride_c,
                                                                   d,
                                                                   ldd,
                                                                   stride_d,
                                                                   batch_count);
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r &&
            c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r &&
            compute_type == rocblas_datatype_i32_r)
    {
        // For now, K must be a multiple of 4, and/or LDA/LDB based on transpose mode
        if(k % 4 != 0 || (trans_a == rocblas_operation_transpose && lda % 4 != 0) ||
           (trans_b == rocblas_operation_none && ldb % 4 != 0))
        {
            rb_status = rocblas_status_invalid_size;
        }
        else
        {
            // adjust by 4 for Tensile
            lda      = (trans_a == rocblas_operation_none) ? lda : lda / 4;
            ldb      = (trans_b == rocblas_operation_none) ? ldb / 4 : ldb;
            stride_a = stride_a / 4;
            stride_b = stride_b / 4;
            k        = k / 4;

            rb_status = gemm_ex_typecasting<TensileInt8x4, TensileInt32, TensileInt32>(handle,
                                                                                       trans_a,
                                                                                       trans_b,
                                                                                       m,
                                                                                       n,
                                                                                       k,
                                                                                       alpha,
                                                                                       a,
                                                                                       lda,
                                                                                       stride_a,
                                                                                       b,
                                                                                       ldb,
                                                                                       stride_b,
                                                                                       beta,
                                                                                       c,
                                                                                       ldc,
                                                                                       stride_c,
                                                                                       d,
                                                                                       ldd,
                                                                                       stride_d,
                                                                                       batch_count);
        }
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}

/*! \brief BLAS EX API

    \details
    GEMM_STRIDED_BATCHED_EX performs one of the strided_batched matrix-matrix operations

        D[i*stride_d] = alpha*op(A[i*stride_a])*op(B[i*stride_b]) + beta*C[i*stride_c], for i in
   [0,batch_count-1]

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, C, and D are strided_batched matrices, with
    op( A ) an m by k by batch_count strided_batched matrix,
    op( B ) a k by n by batch_count strided_batched matrix and
    C and D are m by n by batch_count strided_batched matrices.

    The strided_batched matrices are multiple matrices separated by a constant stride.
    The number of matrices is batch_count.

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
              matrix dimension m
    @param[in]
    n         rocblas_int.
              matrix dimension n
    @param[in]
    k         rocblas_int.
              matrix dimension k
    @param[in]
    alpha     const void *
              specifies the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         void *
              pointer storing matrix A on the GPU.
    @param[in]
    a_type    rocblas_datatype
              specifies the datatype of matrix A
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    stride_a  rocblas_long
              specifies stride from start of one "A" matrix to the next
    @param[in]
    b         void *
              pointer storing matrix B on the GPU.
    @param[in]
    b_type    rocblas_datatype
              specifies the datatype of matrix B
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in]
    stride_b  rocblas_long
              specifies stride from start of one "B" matrix to the next
    @param[in]
    beta      const void *
              specifies the scalar beta. Same datatype as compute_type.
    @param[in]
    c         void *
              pointer storing matrix C on the GPU.
    @param[in]
    c_type    rocblas_datatype
              specifies the datatype of matrix C
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.
    @param[in]
    stride_c  rocblas_long
              specifies stride from start of one "C" matrix to the next
    @param[out]
    d         void *
              pointer storing matrix D on the GPU.
    @param[in]
    d_type    rocblas_datatype
              specifies the datatype of matrix D
    @param[in]
    ldd       rocblas_int
              specifies the leading dimension of D.
    @param[in]
    stride_d  rocblas_long
              specifies stride from start of one "D" matrix to the next
    @param[in]
    batch_count
              rocblas_int
              number of gemm operations in the batch
    @param[in]
    compute_type
              rocblas_datatype
              specifies the datatype of computation
    @param[in]
    algo      rocblas_gemm_algo
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              int32_t
              reserved for future use
    @param[in]
    flags     uint32_t
              reserved for future use
    @param[in/out]
    workspace_size
              size_t*
              size of workspace
    @parm[in]
    workspace void*
              workspace

    ********************************************************************/

extern "C" rocblas_status rocblas_gemm_strided_batched_ex(rocblas_handle handle,
                                                          rocblas_operation trans_a,
                                                          rocblas_operation trans_b,
                                                          rocblas_int m,
                                                          rocblas_int n,
                                                          rocblas_int k,
                                                          const void* alpha,
                                                          const void* a,
                                                          rocblas_datatype a_type,
                                                          rocblas_int lda,
                                                          rocblas_long stride_a,
                                                          const void* b,
                                                          rocblas_datatype b_type,
                                                          rocblas_int ldb,
                                                          rocblas_long stride_b,
                                                          const void* beta,
                                                          const void* c,
                                                          rocblas_datatype c_type,
                                                          rocblas_int ldc,
                                                          rocblas_long stride_c,
                                                          void* d,
                                                          rocblas_datatype d_type,
                                                          rocblas_int ldd,
                                                          rocblas_long stride_d,
                                                          rocblas_int batch_count,
                                                          rocblas_datatype compute_type,
                                                          rocblas_gemm_algo algo,
                                                          int32_t solution_index,
                                                          uint32_t flags,
                                                          size_t* workspace_size,
                                                          void* workspace)
{
    // handle, alpha, beta must not be null pointers for logging
    if(!handle)
        return rocblas_status_invalid_handle;

    if(!alpha || !beta)
        return rocblas_status_invalid_pointer;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        char trans_a_letter, trans_b_letter;
        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            trans_a_letter = rocblas_transpose_letter(trans_a);
            trans_b_letter = rocblas_transpose_letter(trans_b);
        }
        auto a_type_string       = rocblas_datatype_string(a_type);
        auto b_type_string       = rocblas_datatype_string(b_type);
        auto c_type_string       = rocblas_datatype_string(c_type);
        auto d_type_string       = rocblas_datatype_string(d_type);
        auto compute_type_string = rocblas_datatype_string(compute_type);

        if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench))
        {
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                double alpha_double;
                double beta_double;
                if(compute_type == rocblas_datatype_f16_r)
                {
                    alpha_double = *static_cast<const _Float16*>(alpha);
                    beta_double  = *static_cast<const _Float16*>(beta);
                }
                else if(compute_type == rocblas_datatype_f32_r)
                {
                    alpha_double = *static_cast<const float*>(alpha);
                    beta_double  = *static_cast<const float*>(beta);
                }
                else if(compute_type == rocblas_datatype_f64_r)
                {
                    alpha_double = *static_cast<const double*>(alpha);
                    beta_double  = *static_cast<const double*>(beta);
                }
                else if(compute_type == rocblas_datatype_i32_r)
                {
                    alpha_double = *static_cast<const int32_t*>(alpha);
                    beta_double  = *static_cast<const int32_t*>(beta);
                }
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    log_trace(handle,
                              "rocblas_gemm_strided_batched_ex",
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              alpha_double,
                              a,
                              a_type_string,
                              lda,
                              stride_a,
                              b,
                              b_type_string,
                              ldb,
                              stride_b,
                              beta_double,
                              c,
                              c_type_string,
                              ldc,
                              stride_c,
                              d,
                              d_type_string,
                              ldd,
                              stride_d,
                              batch_count,
                              compute_type_string,
                              algo,
                              solution_index,
                              flags,
                              workspace_size,
                              workspace);
                }
                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench -f gemm_strided_batched_ex",
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
                              alpha_double,
                              "--a_type",
                              a_type_string,
                              "--lda",
                              lda,
                              "--stride_a",
                              stride_a,
                              "--b_type",
                              b_type_string,
                              "--ldb",
                              ldb,
                              "--stride_b",
                              stride_b,
                              "--beta",
                              beta_double,
                              "--c_type",
                              c_type_string,
                              "--ldc",
                              ldc,
                              "--stride_c",
                              stride_c,
                              "--d_type",
                              d_type_string,
                              "--ldd",
                              ldd,
                              "--stride_d",
                              stride_d,
                              "--batch",
                              batch_count,
                              "--compute_type",
                              compute_type_string,
                              "--algo",
                              algo,
                              "--solution_index",
                              solution_index,
                              "--flags",
                              flags,
                              "--workspace_size",
                              workspace_size ? *workspace_size : 0);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    log_trace(handle,
                              "rocblas_gemm_strided_batched_ex",
                              trans_a,
                              trans_b,
                              m,
                              n,
                              k,
                              alpha,
                              a,
                              a_type,
                              lda,
                              stride_a,
                              b,
                              b_type,
                              ldb,
                              stride_b,
                              beta,
                              c,
                              c_type,
                              ldc,
                              stride_c,
                              d,
                              d_type,
                              ldd,
                              stride_d,
                              batch_count,
                              compute_type,
                              algo,
                              solution_index,
                              flags,
                              "--workspace_size",
                              workspace_size ? *workspace_size : 0);
                }
            }
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            log_profile(handle,
                        "rocblas_gemm_strided_batched_ex",
                        "a_type",
                        a_type_string,
                        "b_type",
                        b_type_string,
                        "c_type",
                        c_type_string,
                        "d_type",
                        d_type_string,
                        "compute_type",
                        compute_type_string,
                        "transA",
                        trans_a_letter,
                        "transB",
                        trans_b_letter,
                        "M",
                        m,
                        "N",
                        n,
                        "K",
                        k,
                        "lda",
                        lda,
                        "stride_a",
                        stride_a,
                        "ldb",
                        ldb,
                        "stride_b",
                        stride_b,
                        "ldc",
                        ldc,
                        "stride_c",
                        stride_c,
                        "ldd",
                        ldd,
                        "stride_d",
                        stride_d,
                        "batch_count",
                        batch_count,
                        "algo",
                        algo,
                        "solution_index",
                        solution_index,
                        "flags",
                        flags,
                        "workspace_size",
                        workspace_size ? *workspace_size : 0);
        }
    }

    // quick return m,n,k equal to 0 is valid in BLAS
    if(!m || !n || !k || !batch_count)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // pointers must be valid
    if(!a || !b || !c || !d)
        return rocblas_status_invalid_pointer;

    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? m : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;
    rocblas_int num_rows_c = m;
    rocblas_int num_rows_d = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc || num_rows_d > ldd)
        return rocblas_status_invalid_size;

    rocblas_status rb_status = rocblas_status_internal_error;

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r &&
       c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r &&
       compute_type == rocblas_datatype_f64_r)
    {
        rb_status = gemm_ex_typecasting<double, double, double>(handle,
                                                                trans_a,
                                                                trans_b,
                                                                m,
                                                                n,
                                                                k,
                                                                alpha,
                                                                a,
                                                                lda,
                                                                stride_a,
                                                                b,
                                                                ldb,
                                                                stride_b,
                                                                beta,
                                                                c,
                                                                ldc,
                                                                stride_c,
                                                                d,
                                                                ldd,
                                                                stride_d,
                                                                batch_count);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r &&
            c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r &&
            compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<float, float, float>(handle,
                                                             trans_a,
                                                             trans_b,
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             a,
                                                             lda,
                                                             stride_a,
                                                             b,
                                                             ldb,
                                                             stride_b,
                                                             beta,
                                                             c,
                                                             ldc,
                                                             stride_c,
                                                             d,
                                                             ldd,
                                                             stride_d,
                                                             batch_count);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
            c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r &&
            compute_type == rocblas_datatype_f16_r)
    {
        rb_status = gemm_ex_typecasting<_Float16, _Float16, _Float16>(handle,
                                                                      trans_a,
                                                                      trans_b,
                                                                      m,
                                                                      n,
                                                                      k,
                                                                      alpha,
                                                                      a,
                                                                      lda,
                                                                      stride_a,
                                                                      b,
                                                                      ldb,
                                                                      stride_b,
                                                                      beta,
                                                                      c,
                                                                      ldc,
                                                                      stride_c,
                                                                      d,
                                                                      ldd,
                                                                      stride_d,
                                                                      batch_count);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
            c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r &&
            compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<_Float16, _Float16, float>(handle,
                                                                   trans_a,
                                                                   trans_b,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   alpha,
                                                                   a,
                                                                   lda,
                                                                   stride_a,
                                                                   b,
                                                                   ldb,
                                                                   stride_b,
                                                                   beta,
                                                                   c,
                                                                   ldc,
                                                                   stride_c,
                                                                   d,
                                                                   ldd,
                                                                   stride_d,
                                                                   batch_count);
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r &&
            c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r &&
            compute_type == rocblas_datatype_i32_r)
    {
        // For now, K must be a multiple of 4
        if(k % 4 != 0 || ((trans_a == rocblas_operation_transpose) && (lda % 4 != 0)) ||
           ((trans_b == rocblas_operation_none) && (ldb % 4 != 0)) || stride_a % 4 != 0 ||
           stride_b % 4 != 0)
        {
            rb_status = rocblas_status_invalid_size;
        }
        else
        {
            // adjust by 4 for Tensile
            lda      = (trans_a == rocblas_operation_none) ? lda : lda / 4;
            ldb      = (trans_b == rocblas_operation_none) ? ldb / 4 : ldb;
            stride_a = stride_a / 4;
            stride_b = stride_b / 4;
            k        = k / 4;

            rb_status = gemm_ex_typecasting<TensileInt8x4, TensileInt32, TensileInt32>(handle,
                                                                                       trans_a,
                                                                                       trans_b,
                                                                                       m,
                                                                                       n,
                                                                                       k,
                                                                                       alpha,
                                                                                       a,
                                                                                       lda,
                                                                                       stride_a,
                                                                                       b,
                                                                                       ldb,
                                                                                       stride_b,
                                                                                       beta,
                                                                                       c,
                                                                                       ldc,
                                                                                       stride_c,
                                                                                       d,
                                                                                       ldd,
                                                                                       stride_d,
                                                                                       batch_count);
        }
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}
