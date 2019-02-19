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
#include "rocblas_trsm.hpp"

/*! BLAS EX API

    \details
    TRSM_EX solves

        op(A)*X = alpha*B or X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices,
    A is triangular matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    The matrix X is overwritten on B.

    @param[in]
    handle  rocblas_handle.
            handle to the rocblas library context queue.

    @param[in]
    side    rocblas_side.
            rocblas_side_left:       op(A)*X = alpha*B.
            rocblas_side_right:      X*op(A) = alpha*B.

    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a lower triangular matrix.

    @param[in]
    transA  rocblas_operation.
            transB:    op(A) = A.
            rocblas_operation_transpose:      op(A) = A^T.
            rocblas_operation_conjugate_transpose:  op(A) = A^H.

    @param[in]
    diag    rocblas_diagonal.
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       rocblas_int.
            m specifies the number of rows of B. m >= 0.

    @param[in]
    n       rocblas_int.
            n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha
            alpha specifies the scalar alpha. When alpha is
            &zero then A is not referenced, and B need not be set before
            entry.

    @param[in]
    A       void *
            pointer storing matrix A on the GPU.
            of dimension ( lda, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right
            only the upper/lower triangular part is accessed.

    @param[in]
    lda     rocblas_int.
            lda specifies the first dimension of A.
            if side = rocblas_side_left,  lda >= max( 1, m ),
            if side = rocblas_side_right, lda >= max( 1, n ).

    @param[in,output]
    B       void *
            pointer storing matrix B on the GPU.
            B is of dimension ( ldb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    @param[in]
    ldb    rocblas_int.
           ldb specifies the first dimension of B. ldb >= max( 1, m ).

    @param[in]
    invA    void *
            pointer storing the inverse diagonal blocks of A on the GPU.
            invA is of dimension ( ld_invA, k ), where k is m
            when rocblas_side_left and
            is n when rocblas_side_right.

    @param[in]
    ld_invA rocblas_int.
            ldb specifies the first dimension of invA. ld_invA >= max( 1, BLOCK ).

    @param[in]
    compute_type rocblas_datatype
            specifies the datatype of computation

    @param[in]
    option  rocblas_trsm_option
            enumerant specifying the selected trsm memory option.

    @param[in/out]
    x_temp_size size_t*
            During setup the suggested size of x_temp is returned with respect
            to the rocblas_trsm_option specified.
            During run size specifies the size allocated for x_temp_workspace
    @parm[in]
    x_temp_workspace void*
            During setup x_temp_workspace must hold a null pointer
            During run x_temp_workspace is a pointer to store temporary matrix X
            on the GPU.
            x_temp_workspace is of dimension ( x_temp_size/m, x_temp_size/n )

    ********************************************************************/

     rocblas_status rocblas_trsm_ex(rocblas_handle handle,
                                          rocblas_side side,
                                          rocblas_fill uplo,
                                          rocblas_operation trans_a,
                                          rocblas_diagonal diag,
                                          rocblas_int m,
                                          rocblas_int n,
                                          const void* alpha,
                                          const void* a,
                                          rocblas_int lda,
                                          void* b,
                                          rocblas_int ldb,
                                          const void* invA,
                                          rocblas_int ld_invA,
                                          rocblas_datatype compute_type,
                                          rocblas_trsm_option option,
                                          size_t* x_temp_size,
                                          void* x_temp_workspace)

{
    if(!x_temp_workspace)
    {
        if(option == rocblas_trsm_highest_performance)
            *x_temp_size = m*n;
        else if(option == rocblas_trsm_smallest_memory)
            *x_temp_size = m;

        return rocblas_status_success;
    }

    // handle, alpha must not be null pointers for logging
    if(!handle)
        return rocblas_status_invalid_handle;

    if(!alpha)
        return rocblas_status_invalid_pointer;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        auto trans_a_letter = rocblas_transpose_letter(trans_a);
        auto compute_type_string = rocblas_datatype_string(compute_type);

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            double alpha_double;
            if(compute_type == rocblas_datatype_f16_r)
            {
                alpha_double = *static_cast<const _Float16*>(alpha);
            }
            else if(compute_type == rocblas_datatype_f32_r)
            {
                alpha_double = *static_cast<const float*>(alpha);
            }
            else if(compute_type == rocblas_datatype_f64_r)
            {
                alpha_double = *static_cast<const double*>(alpha);
            }
            else if(compute_type == rocblas_datatype_i32_r)
            {
                alpha_double = *static_cast<const int32_t*>(alpha);
            }

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          "rocblas_trsm_ex",
                          trans_a,
                          side,
                          uplo,
                          m,
                          n,
                          alpha_double,
                          a,
                          lda,
                          b,
                          ldb,
                          invA,
                          ld_invA,
                          compute_type,
                          option,
                          x_temp_workspace ? *x_temp_size : 0,
                          x_temp_workspace);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                log_bench(handle,
                          "./rocblas-bench -f trsm_ex",
                          "--transposeA",
                          trans_a_letter,
                          "-m",
                          m,
                          "-n",
                          n,
                          "--alpha",
                          alpha_double,
                          "--lda",
                          lda,
                          "--ldb",
                          ldb,
                          "--ld_invA",
                          ld_invA,
                          "--compute_type",
                          compute_type_string,
                          "--option",
                          option,
                          "--x_temp_size",
                          x_temp_workspace ? *x_temp_size : 0);
            }
        }
        else
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          "rocblas_trsm_ex",
                          trans_a,
                          side,
                          uplo,
                          m,
                          n,
                          alpha,
                          a,
                          lda,
                          b,
                          ldb,
                          invA,
                          ld_invA,
                          x_temp_workspace,
                          compute_type_string,
                          option,
                          "--workspace_size",
                          x_temp_workspace ? *x_temp_size : 0,
                          x_temp_workspace);
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            log_profile(handle,
                        "rocblas_trsm_ex",
                        "compute_type",
                        compute_type_string,
                        "transA",
                        trans_a_letter,
                        rocblas_side_letter(side),
                        side,
                        rocblas_fill_letter(uplo),
                        uplo,
                        "M",
                        m,
                        "N",
                        n,
                        "lda",
                        lda,
                        "ldb",
                        ldb,
                        "ld_invA",
                        ld_invA,
                        "--option",
                        option,
                        "--x_temp_size",
                        x_temp_workspace ? *x_temp_size : 0);
        }
    }

    // quick return m,n,k equal to 0 is valid in BLAS
    if(!m || !n )
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 )
        return rocblas_status_invalid_size;

    // pointers must be valid
    if(!a || !b )
        return rocblas_status_invalid_pointer;

    rocblas_int num_rows_a = side == rocblas_side_left ? m : n;
    rocblas_int num_rows_b = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb)
        return rocblas_status_invalid_size;

    rocblas_status rb_status = rocblas_status_internal_error;


    if(compute_type == rocblas_datatype_f64_r)
    {
        static constexpr rocblas_int DTRSM_BLOCK = 128;
        rb_status = rocblas_trsm_ex_template<DTRSM_BLOCK>(handle,
                                                    side,
                                                    uplo,
                                                    trans_a,
                                                    diag,
                                                    m,
                                                    n,
                                                    static_cast<const double*>(alpha),
                                                    static_cast<const double*>(a),
                                                    lda,
                                                    static_cast<double*>(b),
                                                    ldb,
                                                    static_cast<const double*>(invA),
                                                    ld_invA,
                                                    static_cast<double*>(x_temp_workspace));
    }
    else if(compute_type == rocblas_datatype_f32_r)
    {
        static constexpr rocblas_int STRSM_BLOCK = 128;
        rb_status = rocblas_trsm_ex_template<STRSM_BLOCK>(handle,
                                                    side,
                                                    uplo,
                                                    trans_a,
                                                    diag,
                                                    m,
                                                    n,
                                                    static_cast<const float*>(alpha),
                                                    static_cast<const float*>(a),
                                                    lda,
                                                    static_cast<float*>(b),
                                                    ldb,
                                                    static_cast<const float*>(invA),
                                                    ld_invA,
                                                    static_cast<float*>(x_temp_workspace));
    }

    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}
