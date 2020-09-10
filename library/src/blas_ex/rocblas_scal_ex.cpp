/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_scal_ex.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"

namespace
{
    template <rocblas_int NB>
    rocblas_status rocblas_scal_ex_impl(rocblas_handle   handle,
                                        rocblas_int      n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        void*            x,
                                        rocblas_datatype x_type,
                                        rocblas_int      incx,
                                        rocblas_datatype execution_type)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto alpha_type_str = rocblas_datatype_string(alpha_type);
            auto x_type_str     = rocblas_datatype_string(x_type);
            auto ex_type_str    = rocblas_datatype_string(execution_type);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    rocblas_ostream alphass, betass;
                    if(log_trace_alpha_beta_ex(alpha_type, alpha, nullptr, alphass, betass)
                       == rocblas_status_success)
                    {
                        log_trace(handle,
                                  "rocblas_scal_ex",
                                  n,
                                  alphass.str(),
                                  alpha_type_str,
                                  x,
                                  x_type_str,
                                  incx,
                                  ex_type_str);
                    }
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;
                    if(log_bench_alpha_beta_ex(alpha_type, alpha, nullptr, alphas, betas)
                       == rocblas_status_success)
                    {
                        log_bench(handle,
                                  "./rocblas-bench -f scal_ex",
                                  "-n",
                                  n,
                                  alphas,
                                  "--incx",
                                  incx,
                                  log_bench_scal_precisions(alpha_type, x_type, execution_type));
                    }
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              "rocblas_scal_ex",
                              n,
                              alpha_type_str,
                              x,
                              x_type_str,
                              incx,
                              ex_type_str);
            }
            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            "rocblas_scal_ex",
                            "N",
                            n,
                            "a_type",
                            alpha_type_str,
                            "b_type",
                            x_type_str,
                            "incx",
                            incx,
                            "compute_type",
                            ex_type_str);
        }

        static constexpr rocblas_int    batch_count_1 = 1;
        static constexpr rocblas_stride stride_0      = 0;
        return rocblas_scal_ex_template<NB>(
            handle, n, alpha, alpha_type, x, x_type, incx, stride_0, batch_count_1, execution_type);
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_scal_ex(rocblas_handle   handle,
                               rocblas_int      n,
                               const void*      alpha,
                               rocblas_datatype alpha_type,
                               void*            x,
                               rocblas_datatype x_type,
                               rocblas_int      incx,
                               rocblas_datatype execution_type)
try
{
    constexpr rocblas_int NB = 256;
    return rocblas_scal_ex_impl<NB>(handle, n, alpha, alpha_type, x, x_type, incx, execution_type);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
