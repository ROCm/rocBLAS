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

extern "C" rocblas_status rocblas_trsm_ex(rocblas_handle handle,
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
    // handle, alpha must not be null pointers for logging
    if(!handle)
        return rocblas_status_invalid_handle;

    static constexpr rocblas_int TRSM_BLOCK = 128;
    rocblas_int k                           = (side == rocblas_side_left ? m : n);
    bool allowChunking = (k % TRSM_BLOCK == 0 && k <= TRSM_BLOCK * *(handle->get_trsm_A_blks()));

    if(!x_temp_workspace)
    {
        rocblas_int k = (side == rocblas_side_left ? m : n);

        if(option == rocblas_trsm_high_performance || !allowChunking)
            *x_temp_size = m * n;
        else if(option == rocblas_trsm_low_memory)
            *x_temp_size = m;
        else
            return rocblas_status_not_implemented;

        return rocblas_status_success;
    }

    if(!alpha)
        return rocblas_status_invalid_pointer;

    if(!allowChunking && (*x_temp_size / m) < n) // Chunking not supported
        return rocblas_status_invalid_size;

    auto layer_mode = handle->layer_mode;
    if(layer_mode & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench |
                     rocblas_layer_mode_log_profile))
    {
        auto trans_a_letter      = rocblas_transpose_letter(trans_a);
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
    if(!m || !n)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0)
        return rocblas_status_invalid_size;

    // pointers must be valid
    if(!a || !b)
        return rocblas_status_invalid_pointer;

    rocblas_int num_rows_a = side == rocblas_side_left ? m : n;
    rocblas_int num_rows_b = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || ld_invA != TRSM_BLOCK)
        return rocblas_status_invalid_size;

    rocblas_status rb_status = rocblas_status_internal_error;

    if(compute_type == rocblas_datatype_f64_r)
    {
        rb_status = rocblas_trsm_ex_template<TRSM_BLOCK>(handle,
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
                                                         static_cast<const size_t*>(x_temp_size),
                                                         static_cast<double*>(x_temp_workspace));
    }
    else if(compute_type == rocblas_datatype_f32_r)
    {
        rb_status = rocblas_trsm_ex_template<TRSM_BLOCK>(handle,
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
                                                         static_cast<const size_t*>(x_temp_size),
                                                         static_cast<float*>(x_temp_workspace));
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}
