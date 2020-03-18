/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_gemm_ext2.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

// This functionality is only availble when using the new Tensile client
#ifdef USE_TENSILE_HOST

rocblas_status rocblas_gemm_ext2_impl(rocblas_handle    handle,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const void*       alpha,
                                      const void*       a,
                                      rocblas_datatype  a_type,
                                      rocblas_int       row_stride_a,
                                      rocblas_int       col_stride_a,
                                      const void*       b,
                                      rocblas_datatype  b_type,
                                      rocblas_int       row_stride_b,
                                      rocblas_int       col_stride_b,
                                      const void*       beta,
                                      const void*       c,
                                      rocblas_datatype  c_type,
                                      rocblas_int       row_stride_c,
                                      rocblas_int       col_stride_c,
                                      void*             d,
                                      rocblas_datatype  d_type,
                                      rocblas_int       row_stride_d,
                                      rocblas_int       col_stride_d,
                                      rocblas_datatype  compute_type,
                                      rocblas_gemm_algo algo,
                                      int32_t           solution_index,
                                      uint32_t          flags)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

#if 0 // TODO: Implement logging with correct argument names

    auto layer_mode = handle->layer_mode;
    if(layer_mode
       & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
          | rocblas_layer_mode_log_profile))
    {
        auto a_type_string       = rocblas_datatype_string(a_type);
        auto b_type_string       = rocblas_datatype_string(b_type);
        auto c_type_string       = rocblas_datatype_string(c_type);
        auto d_type_string       = rocblas_datatype_string(d_type);
        auto compute_type_string = rocblas_datatype_string(compute_type);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_trace))
        {
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    std::stringstream alphass, betass;
                    if(log_trace_alpha_beta_ex(compute_type, alpha, beta, alphass, betass)
                       == rocblas_status_success)
                    {
                        log_trace(handle,
                                  "rocblas_gemm_ext2",
                                  m,
                                  n,
                                  k,
                                  alphass.str(),
                                  a,
                                  a_type_string,
                                  row_stride_a,
                                  col_stride_a,
                                  b,
                                  b_type_string,
                                  row_stride_b,
                                  b_col_stide,
                                  betass.str(),
                                  c,
                                  c_type_string,
                                  c_row_stide,
                                  c_col_stide,
                                  d,
                                  d_type_string,
                                  row_stride_d,
                                  d_col_stide,
                                  compute_type_string,
                                  algo,
                                  solution_index,
                                  flags);
                    }
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;
                    if(log_bench_alpha_beta_ex(compute_type, alpha, beta, alphas, betas)
                       == rocblas_status_success)
                    {
                        log_bench(handle,
                                  "./rocblas-bench -f gemm_ex",
                                  "-m",
                                  m,
                                  "-n",
                                  n,
                                  "-k",
                                  k,
                                  alphas,
                                  "--a_type",
                                  a_type_string,
                                  "--row_stride_a",
                                  row_stride_a,
                                  "--col_stride_a",
                                  col_stride_a,
                                  "--b_type",
                                  b_type_string,
                                  "--row_stride_b",
                                  row_stride_b,
                                  "--col_stride_b",
                                  col_stride_b,
                                  betas,
                                  "--c_type",
                                  c_type_string,
                                  "--row_stride_c",
                                  row_stride_c,
                                  "--col_stride_c",
                                  col_stride_c,
                                  "--d_type",
                                  d_type_string,
                                  "--row_stride_d",
                                  row_stride_d,
                                  "--col_stride_d",
                                  col_stride_d,
                                  "--compute_type",
                                  compute_type_string,
                                  "--algo",
                                  algo,
                                  "--solution_index",
                                  solution_index,
                                  "--flags",
                                  flags);
                    }
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              "rocblas_gemm_ex",
                              m,
                              n,
                              k,
                              alpha,
                              a,
                              a_type_string,
                              row_stride_a,
                              col_stride_a,
                              b,
                              b_type_string,
                              row_stride_b,
                              col_stride_b,
                              beta,
                              c,
                              c_type_string,
                              row_stride_c,
                              col_stride_c,
                              d,
                              d_type_string,
                              row_stride_d,
                              col_stride_d,
                              compute_type_string,
                              algo,
                              solution_index,
                              flags);
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
                        "M",
                        m,
                        "N",
                        n,
                        "K",
                        k,
                        "row_stride_a",
                        row_stride_a,
                        "col_stride_a",
                        col_stride_a,
                        "row_stride_b",
                        row_stride_b,
                        "col_stride_b",
                        col_stride_b,
                        "row_stride_c",
                        row_stride_c,
                        "col_stride_c",
                        col_stride_c,
                        "row_stride_d",
                        row_stride_d,
                        "col_stride_d",
                        col_stride_d,
                        "algo",
                        algo,
                        "solution_index",
                        solution_index,
                        "flags",
                        flags);
        }
    }

#endif // TODO

    // quick return
    // Note: k==0 is not a quick return, because C still has to be multiplied by beta
    if(!m || !n)
        return rocblas_status_success;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0)
        return rocblas_status_invalid_size;

    // pointers must be valid
    if(!a || !b || !c || !d || !alpha || !beta)
        return rocblas_status_invalid_pointer;

    rocblas_stride batch_stride_a = 0;
    rocblas_stride batch_stride_b = 0;
    rocblas_stride batch_stride_c = 0;
    rocblas_stride batch_stride_d = 0;
    rocblas_int    offset         = 0;
    rocblas_int    batch_count    = 1;

    return rocblas_gemm_ext2_template<false>(handle,
                                             rocblas_operation_none,
                                             rocblas_operation_none,
                                             m,
                                             n,
                                             k,
                                             alpha,
                                             a,
                                             a_type,
                                             offset,
                                             row_stride_a,
                                             col_stride_a,
                                             batch_stride_a,
                                             b,
                                             b_type,
                                             offset,
                                             row_stride_b,
                                             col_stride_b,
                                             batch_stride_b,
                                             beta,
                                             c,
                                             c_type,
                                             offset,
                                             row_stride_c,
                                             col_stride_c,
                                             batch_stride_c,
                                             d,
                                             d_type,
                                             offset,
                                             row_stride_d,
                                             col_stride_d,
                                             batch_stride_d,
                                             batch_count,
                                             compute_type);
}

extern "C" rocblas_status rocblas_gemm_ext2(rocblas_handle    handle,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const void*       alpha,
                                            const void*       a,
                                            rocblas_datatype  a_type,
                                            rocblas_int       row_stride_a,
                                            rocblas_int       col_stride_a,
                                            const void*       b,
                                            rocblas_datatype  b_type,
                                            rocblas_int       row_stride_b,
                                            rocblas_int       col_stride_b,
                                            const void*       beta,
                                            const void*       c,
                                            rocblas_datatype  c_type,
                                            rocblas_int       row_stride_c,
                                            rocblas_int       col_stride_c,
                                            void*             d,
                                            rocblas_datatype  d_type,
                                            rocblas_int       row_stride_d,
                                            rocblas_int       col_stride_d,
                                            rocblas_datatype  compute_type,
                                            rocblas_gemm_algo algo,
                                            int32_t           solution_index,
                                            uint32_t          flags)
try
{
    return rocblas_gemm_ext2_impl(handle,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  a,
                                  a_type,
                                  row_stride_a,
                                  col_stride_a,
                                  b,
                                  b_type,
                                  row_stride_b,
                                  col_stride_b,
                                  beta,
                                  c,
                                  c_type,
                                  row_stride_c,
                                  col_stride_c,
                                  d,
                                  d_type,
                                  row_stride_d,
                                  col_stride_d,
                                  compute_type,
                                  algo,
                                  solution_index,
                                  flags);
}
catch(...)
{
    return exception_to_rocblas_status();
}

#endif // USE_TENSILE_HOST
