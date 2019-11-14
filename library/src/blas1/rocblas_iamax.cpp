/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_iamax.hpp"
#include "rocblas_reduction_impl.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_iamax_name[] = "unknown";
    template <>
    constexpr char rocblas_iamax_name<float>[] = "rocblas_isamax";
    template <>
    constexpr char rocblas_iamax_name<double>[] = "rocblas_idamax";
    template <>
    constexpr char rocblas_iamax_name<rocblas_float_complex>[] = "rocblas_icamax";
    template <>
    constexpr char rocblas_iamax_name<rocblas_double_complex>[] = "rocblas_izamax";

    // allocate workspace inside this API
    template <typename S, typename T>
    rocblas_status rocblas_iamax_impl(
        rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result)
    {
        static constexpr bool           isbatched     = false;
        static constexpr rocblas_stride stridex_0     = 0;
        static constexpr rocblas_int    batch_count_1 = 1;
        static constexpr int            NB            = 1024;

        return rocblas_reduction_impl<NB,
                                      isbatched,
                                      rocblas_fetch_amax_amin<S>,
                                      rocblas_reduce_amax,
                                      rocblas_finalize_amax_amin,
                                      index_value_t<S>>(
            handle, n, x, incx, stridex_0, batch_count_1, result, rocblas_iamax_name<T>, "iamax");
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

#define IMPL(name_, typei_, typew_)                                     \
    rocblas_status name_(rocblas_handle handle,                         \
                         rocblas_int    n,                              \
                         const typei_*  x,                              \
                         rocblas_int    incx,                           \
                         rocblas_int*   results)                        \
    {                                                                   \
        return rocblas_iamax_impl<typew_>(handle, n, x, incx, results); \
    }

IMPL(rocblas_isamax, float, float);
IMPL(rocblas_idamax, double, double);
IMPL(rocblas_icamax, rocblas_float_complex, float);
IMPL(rocblas_izamax, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
