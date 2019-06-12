/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "Tensile.h"
#include "TensileTypes.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_trsm.hpp"
#include "status.h"
#include "utility.h"
#include <type_traits>

constexpr size_t WORKBUF_TRSM_A_BLKS     = 10;
constexpr size_t WORKBUF_TRSM_B_MIN_CHNK = 1024;
constexpr size_t WORKBUF_TRSM_INVA_SZ    = 128 * 128 * 10 * sizeof(double);
constexpr size_t WORKBUF_TRSM_INVA_C_SZ  = 128 * 128 * 10 * sizeof(double) / 2;
constexpr size_t WORKBUF_TRSV_X_SZ       = 131072 * sizeof(double);
constexpr size_t WORKBUF_TRSV_ALPHA_SZ   = sizeof(double);

extern "C" rocblas_status rocblas_trsm_ex(rocblas_handle    handle,
                                          rocblas_side      side,
                                          rocblas_fill      uplo,
                                          rocblas_operation trans_a,
                                          rocblas_diagonal  diag,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          const void*       alpha,
                                          const void*       a,
                                          rocblas_int       lda,
                                          void*             b,
                                          rocblas_int       ldb,
                                          const void*       invA,
                                          rocblas_int       ld_invA,
                                          rocblas_datatype  compute_type)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // Compute the optimum size in bytes for maximum speed
    size_t x_temp_size = rocblas_sizeof_datatype(compute_type) * m * n;

    // If this call is a device memory size query,
    if(handle->is_device_memory_size_query())
    {
        // return the size in bytes recommended for maximum speed
        return handle->set_optimal_device_memory_size(x_temp_size);
    }

    static constexpr rocblas_int TRSM_BLOCK = 128;
    rocblas_int                  k          = (side == rocblas_side_left ? m : n);

    // Attempt to allocate the optimal size
    void* x_temp_workspace = handle->device_memory_alloc(x_temp_size);

    // If optimal size is not available, try the smaller size
    if(!x_temp_workspace)
    {
        bool allowChunking = (k % TRSM_BLOCK == 0 && k <= TRSM_BLOCK * WORKBUF_TRSM_A_BLKS);

        if(!allowChunking) // Chunking not supported
            return rocblas_status_memory_error;

        x_temp_size = rocblas_sizeof_datatype(compute_type) * m;
        //        x_temp_workspace = handle->device_memory_alloc(x_temp_size);

        // If the smaller size cannot be allocated, return error
        if(!x_temp_workspace)
            return rocblas_status_memory_error;
    }

    if(!alpha)
        return rocblas_status_invalid_pointer;

    auto layer_mode = handle->layer_mode;
    if(layer_mode
       & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
          | rocblas_layer_mode_log_profile))
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
                          compute_type);

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
                          compute_type_string);
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
                          compute_type_string);
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
                        ld_invA);
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
                                                         &x_temp_size,
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
                                                         &x_temp_size,
                                                         static_cast<float*>(x_temp_workspace));
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}
