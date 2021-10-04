/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "logging.hpp"
#include "rocblas_hemv_symv.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_symv_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_symv_batched_name<float>[] = "rocblas_ssymv_batched";
    template <>
    constexpr char rocblas_symv_batched_name<double>[] = "rocblas_dsymv_batched";
    template <>
    constexpr char rocblas_symv_batched_name<rocblas_float_complex>[] = "rocblas_csymv_batched";
    template <>
    constexpr char rocblas_symv_batched_name<rocblas_double_complex>[] = "rocblas_zsymv_batched";

    template <typename T, typename U, typename V, typename W>
    rocblas_status rocblas_symv_batched_impl(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             const V*       alpha,
                                             const U*       A,
                                             rocblas_int    lda,
                                             const U*       x,
                                             rocblas_int    incx,
                                             const V*       beta,
                                             W*             y,
                                             rocblas_int    incy,
                                             rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto check_numerics = handle->check_numerics;

        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto uplo_letter = rocblas_fill_letter(uplo);

                if(layer_mode & rocblas_layer_mode_log_trace)
                    log_trace(handle,
                              rocblas_symv_batched_name<T>,
                              uplo,
                              n,
                              LOG_TRACE_SCALAR_VALUE(handle, alpha),
                              A,
                              lda,
                              x,
                              incx,
                              LOG_TRACE_SCALAR_VALUE(handle, beta),
                              y,
                              incy,
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    log_bench(handle,
                              "./rocblas-bench -f symv_batched -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "-n",
                              n,
                              LOG_BENCH_SCALAR_VALUE(handle, alpha),
                              "--lda",
                              lda,
                              "--incx",
                              incx,
                              LOG_BENCH_SCALAR_VALUE(handle, beta),
                              "--incy",
                              incy,
                              "--batch_count",
                              batch_count);

                if(layer_mode & rocblas_layer_mode_log_profile)
                    log_profile(handle,
                                rocblas_symv_batched_name<T>,
                                "uplo",
                                uplo_letter,
                                "N",
                                n,
                                "lda",
                                lda,
                                "incx",
                                incx,
                                "incy",
                                incy,
                                "batch_count",
                                batch_count);
            }
        }

        rocblas_status arg_status = rocblas_symv_arg_check<T>(handle,
                                                              uplo,
                                                              n,
                                                              alpha,
                                                              0,
                                                              A,
                                                              0,
                                                              lda,
                                                              0,
                                                              x,
                                                              0,
                                                              incx,
                                                              0,
                                                              beta,
                                                              0,
                                                              y,
                                                              0,
                                                              incy,
                                                              0,
                                                              batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        //allocating the workspace identical to hemv
        size_t dev_bytes = rocblas_internal_hemv_symv_kernel_workspace_size<T>(n, batch_count);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto workspace = handle->device_malloc(dev_bytes);
        if(!workspace)
            return rocblas_status_memory_error;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status symv_check_numerics_status
                = rocblas_symv_check_numerics(rocblas_symv_batched_name<T>,
                                              handle,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              y,
                                              0,
                                              incy,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(symv_check_numerics_status != rocblas_status_success)
                return symv_check_numerics_status;
        }

        rocblas_status status = rocblas_internal_symv_template<T>(handle,
                                                                  uplo,
                                                                  n,
                                                                  alpha,
                                                                  0,
                                                                  A,
                                                                  0,
                                                                  lda,
                                                                  0,
                                                                  x,
                                                                  0,
                                                                  incx,
                                                                  0,
                                                                  beta,
                                                                  0,
                                                                  y,
                                                                  0,
                                                                  incy,
                                                                  0,
                                                                  batch_count,
                                                                  (V*)workspace);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status symv_check_numerics_status
                = rocblas_symv_check_numerics(rocblas_symv_batched_name<T>,
                                              handle,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              y,
                                              0,
                                              incy,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(symv_check_numerics_status != rocblas_status_success)
                return symv_check_numerics_status;
        }
        return status;
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, T_)                                                   \
    rocblas_status routine_name_(rocblas_handle  handle,                          \
                                 rocblas_fill    uplo,                            \
                                 rocblas_int     n,                               \
                                 const T_*       alpha,                           \
                                 const T_* const A[],                             \
                                 rocblas_int     lda,                             \
                                 const T_* const x[],                             \
                                 rocblas_int     incx,                            \
                                 const T_*       beta,                            \
                                 T_* const       y[],                             \
                                 rocblas_int     incy,                            \
                                 rocblas_int     batch_count)                     \
    try                                                                           \
    {                                                                             \
        return rocblas_symv_batched_impl<T_>(                                     \
            handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy, batch_count); \
    }                                                                             \
    catch(...)                                                                    \
    {                                                                             \
        return exception_to_rocblas_status();                                     \
    }

IMPL(rocblas_ssymv_batched, float);
IMPL(rocblas_dsymv_batched, double);
IMPL(rocblas_csymv_batched, rocblas_float_complex);
IMPL(rocblas_zsymv_batched, rocblas_double_complex);

#undef IMPL

} // extern "C"
