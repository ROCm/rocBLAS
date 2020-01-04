/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_asum.hpp"
#include "rocblas_reduction_impl.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_asum_name[] = "unknown";
    template <>
    constexpr char rocblas_asum_name<float>[] = "rocblas_sasum";
    template <>
    constexpr char rocblas_asum_name<double>[] = "rocblas_dasum";
    template <>
    constexpr char rocblas_asum_name<rocblas_float_complex>[] = "rocblas_scasum";
    template <>
    constexpr char rocblas_asum_name<rocblas_double_complex>[] = "rocblas_dzasum";

    // allocate workspace inside this API
    template <rocblas_int NB, typename Ti, typename To>
    rocblas_status rocblas_asum_impl(
        rocblas_handle handle, rocblas_int n, const Ti* x, rocblas_int incx, To* results)
    {
        static constexpr bool           isbatched     = false;
        static constexpr rocblas_stride stridex_0     = 0;
        static constexpr rocblas_int    batch_count_1 = 1;

        return rocblas_reduction_impl<NB,
                                      isbatched,
                                      rocblas_fetch_asum<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_identity,
                                      To>(
            handle, n, x, incx, stridex_0, batch_count_1, results, rocblas_asum_name<Ti>, "asum");
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL IS ALREADY DEFINED
#endif

#define IMPL(name_, typei_, typeo_)                                                               \
    rocblas_status name_(                                                                         \
        rocblas_handle handle, rocblas_int n, const typei_* x, rocblas_int incx, typeo_* results) \
    try                                                                                           \
    {                                                                                             \
        constexpr rocblas_int NB = 512;                                                           \
        return rocblas_asum_impl<NB>(handle, n, x, incx, results);                                \
    }                                                                                             \
    catch(...)                                                                                    \
    {                                                                                             \
        return exception_to_rocblas_status();                                                     \
    }

IMPL(rocblas_sasum, float, float);
IMPL(rocblas_dasum, double, double);
IMPL(rocblas_scasum, rocblas_float_complex, float);
IMPL(rocblas_dzasum, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
