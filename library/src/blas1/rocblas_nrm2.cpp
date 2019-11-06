/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_nrm2.hpp"
#include "rocblas_reduction_impl.hpp"

namespace
{

    template <typename>
    constexpr char rocblas_nrm2_name[] = "unknown";
    template <>
    constexpr char rocblas_nrm2_name<float>[] = "rocblas_snrm2";
    template <>
    constexpr char rocblas_nrm2_name<double>[] = "rocblas_dnrm2";
    template <>
    constexpr char rocblas_nrm2_name<rocblas_half>[] = "rocblas_hnrm2";
    template <>
    constexpr char rocblas_nrm2_name<rocblas_float_complex>[] = "rocblas_scnrm2";
    template <>
    constexpr char rocblas_nrm2_name<rocblas_double_complex>[] = "rocblas_dznrm2";

    // allocate workspace inside this API
    template <rocblas_int NB, typename Ti, typename To>
    rocblas_status rocblas_nrm2_impl(
        rocblas_handle handle, rocblas_int n, const Ti* x, rocblas_int incx, To* results)
    {
        static constexpr bool           isbatched     = false;
        static constexpr rocblas_stride stridex_0     = 0;
        static constexpr rocblas_int    batch_count_1 = 1;
        return rocblas_reduction_impl<NB,
                                      isbatched,
                                      rocblas_fetch_nrm2<To>,
                                      rocblas_reduce_sum,
                                      rocblas_finalize_nrm2,
                                      To>(
            handle, n, x, incx, stridex_0, batch_count_1, results, rocblas_nrm2_name<Ti>, "nrm2");
    }

} // namespace

/* ============================================================================================ */

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
    {                                                                                             \
        constexpr rocblas_int NB = 512;                                                           \
        return rocblas_nrm2_impl<NB>(handle, n, x, incx, results);                                \
    }

IMPL(rocblas_snrm2, float, float);
IMPL(rocblas_dnrm2, double, double);
IMPL(rocblas_scnrm2, rocblas_float_complex, float);
IMPL(rocblas_dznrm2, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
