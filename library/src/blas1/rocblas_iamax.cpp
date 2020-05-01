/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
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
        static constexpr rocblas_int    shiftx_0      = 0;
        static constexpr rocblas_stride stridex_0     = 0;
        static constexpr rocblas_int    batch_count_1 = 1;
        static constexpr int            NB            = 1024;

        rocblas_index_value_t<S>* mem           = nullptr;
        rocblas_status            checks_status = rocblas_reduction_setup<NB, isbatched>(handle,
                                                                              n,
                                                                              x,
                                                                              incx,
                                                                              stridex_0,
                                                                              batch_count_1,
                                                                              result,
                                                                              rocblas_iamax_name<S>,
                                                                              "iamax",
                                                                              mem);
        if(checks_status != rocblas_status_continue)
        {
            return checks_status;
        }

        return rocblas_iamax_template<NB, isbatched>(
            handle, n, x, shiftx_0, incx, stridex_0, batch_count_1, result, mem);
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
    try                                                                 \
    {                                                                   \
        return rocblas_iamax_impl<typew_>(handle, n, x, incx, results); \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

IMPL(rocblas_isamax, float, float);
IMPL(rocblas_idamax, double, double);
IMPL(rocblas_icamax, rocblas_float_complex, float);
IMPL(rocblas_izamax, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
