/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.hpp"
#include "rocblas_axpy_ex.hpp"

namespace
{
    template <int NB>
    rocblas_status rocblas_axpy_batched_ex_impl(rocblas_handle   handle,
                                                rocblas_int      n,
                                                const void*      alpha,
                                                rocblas_datatype alpha_type,
                                                const void*      x,
                                                rocblas_datatype x_type,
                                                rocblas_int      incx,
                                                void*            y,
                                                rocblas_datatype y_type,
                                                rocblas_int      incy,
                                                rocblas_int      batch_count,
                                                rocblas_datatype execution_type,
                                                const char*      name,
                                                const char*      bench_name)
    {
        if(!handle)
        {
            return rocblas_status_invalid_handle;
        }

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto alpha_type_str = rocblas_datatype_string(alpha_type);
            auto x_type_str     = rocblas_datatype_string(x_type);
            auto y_type_str     = rocblas_datatype_string(y_type);
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
                                  name,
                                  n,
                                  alphass.str(),
                                  alpha_type_str,
                                  x,
                                  x_type_str,
                                  incx,
                                  y,
                                  y_type_str,
                                  incy,
                                  batch_count,
                                  ex_type_str);
                    }
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;
                    if(log_bench_alpha_beta_ex(execution_type, alpha, nullptr, alphas, betas)
                       == rocblas_status_success)
                    {
                        log_bench(handle,
                                  "./rocblas-bench",
                                  "-f",
                                  bench_name,
                                  "-n",
                                  n,
                                  alphas,
                                  "--a_type",
                                  alpha_type_str,
                                  "--b_type",
                                  x_type_str,
                                  "--incx",
                                  incx,
                                  "--c_type",
                                  y_type_str,
                                  "--incy",
                                  incy,
                                  "--batch_count",
                                  batch_count,
                                  "--compute_type",
                                  ex_type_str);
                    }
                }
            }
            else if(layer_mode & rocblas_layer_mode_log_trace)
            {
                log_trace(handle,
                          name,
                          n,
                          alpha_type_str,
                          x,
                          x_type_str,
                          incx,
                          y,
                          y_type_str,
                          incy,
                          batch_count,
                          ex_type_str);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                log_profile(handle,
                            name,
                            "N",
                            n,
                            "a_type",
                            alpha_type_str,
                            "b_type",
                            x_type_str,
                            "incx",
                            incx,
                            "c_type",
                            y_type_str,
                            "incy",
                            incy,
                            "batch_count",
                            batch_count,
                            "compute_type",
                            ex_type_str);
            }
        }

        static constexpr rocblas_stride stride_0 = 0;
        return rocblas_axpy_ex_template<NB, true>(handle,
                                                  n,
                                                  alpha,
                                                  alpha_type,
                                                  x,
                                                  x_type,
                                                  incx,
                                                  stride_0,
                                                  y,
                                                  y_type,
                                                  incy,
                                                  stride_0,
                                                  batch_count,
                                                  execution_type);
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_axpy_batched_ex(rocblas_handle   handle,
                                       rocblas_int      n,
                                       const void*      alpha,
                                       rocblas_datatype alpha_type,
                                       const void*      x,
                                       rocblas_datatype x_type,
                                       rocblas_int      incx,
                                       void*            y,
                                       rocblas_datatype y_type,
                                       rocblas_int      incy,
                                       rocblas_int      batch_count,
                                       rocblas_datatype execution_type)
{
    try
    {
        return rocblas_axpy_batched_ex_impl<256>(handle,
                                                 n,
                                                 alpha,
                                                 alpha_type,
                                                 x,
                                                 x_type,
                                                 incx,
                                                 y,
                                                 y_type,
                                                 incy,
                                                 batch_count,
                                                 execution_type,
                                                 "rocblas_axpy_batched_ex",
                                                 "axpy_batched_ex");
    }
    catch(...)
    {
        return exception_to_rocblas_status();
    }
}

} // extern "C"
