/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_iamax_batched.hpp"
#include "rocblas_reduction_impl.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_iamax_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_iamax_batched_name<float>[] = "rocblas_isamax_batched";
    template <>
    constexpr char rocblas_iamax_batched_name<double>[] = "rocblas_idamax_batched";
    template <>
    constexpr char rocblas_iamax_batched_name<rocblas_float_complex>[] = "rocblas_icamax_batched";
    template <>
    constexpr char rocblas_iamax_batched_name<rocblas_double_complex>[] = "rocblas_izamax_batched";

    // allocate workspace inside this API
    template <typename S, typename T>
    rocblas_status rocblas_iamax_batched_impl(rocblas_handle  handle,
                                              rocblas_int     n,
                                              const T* const* x,
                                              rocblas_int     incx,
                                              rocblas_int     batch_count,
                                              rocblas_int*    result)
    {
        static constexpr bool           isbatched = true;
        static constexpr int            NB        = 1024;
        static constexpr rocblas_stride stridex_0 = 0;

        return rocblas_reduction_impl<NB,
                                      isbatched,
                                      rocblas_fetch_amax_amin<S>,
                                      rocblas_reduce_amax,
                                      rocblas_finalize_amax_amin,
                                      index_value_t<S>>(handle,
                                                        n,
                                                        x,
                                                        incx,
                                                        stridex_0,
                                                        batch_count,
                                                        result,
                                                        rocblas_iamax_batched_name<T>,
                                                        "iamax_batched");
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

#define IMPL(name_routine_, T_, S_)                                                      \
    rocblas_status name_routine_(rocblas_handle   handle,                                \
                                 rocblas_int      n,                                     \
                                 const T_* const* x,                                     \
                                 rocblas_int      incx,                                  \
                                 rocblas_int      batch_count,                           \
                                 rocblas_int*     results)                               \
    {                                                                                    \
        return rocblas_iamax_batched_impl<S_>(handle, n, x, incx, batch_count, results); \
    }

IMPL(rocblas_isamax_batched, float, float);
IMPL(rocblas_idamax_batched, double, double);
IMPL(rocblas_icamax_batched, rocblas_float_complex, float);
IMPL(rocblas_izamax_batched, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
