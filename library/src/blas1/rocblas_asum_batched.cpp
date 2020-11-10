/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_asum_batched.hpp"
#include "check_numerics_vector.hpp"
#include "rocblas_reduction_impl.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_asum_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_asum_batched_name<float>[] = "rocblas_sasum_batched";
    template <>
    constexpr char rocblas_asum_batched_name<double>[] = "rocblas_dasum_batched";
    template <>
    constexpr char rocblas_asum_batched_name<rocblas_float_complex>[] = "rocblas_scasum_batched";
    template <>
    constexpr char rocblas_asum_batched_name<rocblas_double_complex>[] = "rocblas_dzasum_batched";

    // allocate workspace inside this API
    template <rocblas_int NB, typename Ti, typename To>
    rocblas_status rocblas_asum_batched_impl(rocblas_handle  handle,
                                             rocblas_int     n,
                                             const Ti* const x[],
                                             rocblas_int     incx,
                                             rocblas_int     batch_count,
                                             To*             results)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        auto                            check_numerics = handle->check_numerics;
        static constexpr bool           isbatched      = true;
        static constexpr rocblas_stride stridex_0      = 0;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status check_numerics_status
                = rocblas_check_numerics_vector_template(rocblas_asum_batched_name<Ti>,
                                                         handle,
                                                         n,
                                                         x,
                                                         0,
                                                         incx,
                                                         stridex_0,
                                                         batch_count,
                                                         check_numerics,
                                                         is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }

        rocblas_status status = rocblas_reduction_impl<NB,
                                                       isbatched,
                                                       rocblas_fetch_asum<To>,
                                                       rocblas_reduce_sum,
                                                       rocblas_finalize_identity,
                                                       To>(handle,
                                                           n,
                                                           x,
                                                           incx,
                                                           stridex_0,
                                                           batch_count,
                                                           results,
                                                           rocblas_asum_batched_name<Ti>,
                                                           "asum_batched");
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status check_numerics_status
                = rocblas_check_numerics_vector_template(rocblas_asum_batched_name<Ti>,
                                                         handle,
                                                         n,
                                                         x,
                                                         0,
                                                         incx,
                                                         stridex_0,
                                                         batch_count,
                                                         check_numerics,
                                                         is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }
        return status;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL IS ALREADY DEFINED
#endif

#define IMPL(name_, typei_, typeo_)                                                    \
    rocblas_status name_(rocblas_handle      handle,                                   \
                         rocblas_int         n,                                        \
                         const typei_* const x[],                                      \
                         rocblas_int         incx,                                     \
                         rocblas_int         batch_count,                              \
                         typeo_*             result)                                   \
    try                                                                                \
    {                                                                                  \
        constexpr rocblas_int NB = 512;                                                \
        return rocblas_asum_batched_impl<NB>(handle, n, x, incx, batch_count, result); \
    }                                                                                  \
    catch(...)                                                                         \
    {                                                                                  \
        return exception_to_rocblas_status();                                          \
    }

IMPL(rocblas_sasum_batched, float, float);
IMPL(rocblas_dasum_batched, double, double);
IMPL(rocblas_scasum_batched, rocblas_float_complex, float);
IMPL(rocblas_dzasum_batched, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
