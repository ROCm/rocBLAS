/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_nrm2.hpp"
#include "check_numerics_vector.hpp"
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
        static constexpr rocblas_int    shiftx_0      = 0;

        size_t         dev_bytes = 0;
        rocblas_status checks_status
            = rocblas_reduction_setup<NB, isbatched, To>(handle,
                                                         n,
                                                         x,
                                                         incx,
                                                         stridex_0,
                                                         batch_count_1,
                                                         results,
                                                         rocblas_nrm2_name<Ti>,
                                                         "nrm2",
                                                         dev_bytes);
        if(checks_status != rocblas_status_continue)
        {
            return checks_status;
        }

        auto check_numerics = handle->check_numerics;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status check_numerics_status
                = rocblas_check_numerics_vector_template(rocblas_nrm2_name<Ti>,
                                                         handle,
                                                         n,
                                                         x,
                                                         0,
                                                         incx,
                                                         stridex_0,
                                                         batch_count_1,
                                                         check_numerics,
                                                         is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }

        auto mem = handle->device_malloc(dev_bytes);
        if(!mem)
        {
            return rocblas_status_memory_error;
        }
        rocblas_status status = rocblas_nrm2_template<NB, isbatched>(
            handle, n, x, shiftx_0, incx, stridex_0, batch_count_1, results, (To*)mem);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status check_numerics_status
                = rocblas_check_numerics_vector_template(rocblas_nrm2_name<Ti>,
                                                         handle,
                                                         n,
                                                         x,
                                                         0,
                                                         incx,
                                                         stridex_0,
                                                         batch_count_1,
                                                         check_numerics,
                                                         is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }
        return status;
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
    try                                                                                           \
    {                                                                                             \
        constexpr rocblas_int NB = 512;                                                           \
        return rocblas_nrm2_impl<NB>(handle, n, x, incx, results);                                \
    }                                                                                             \
    catch(...)                                                                                    \
    {                                                                                             \
        return exception_to_rocblas_status();                                                     \
    }

IMPL(rocblas_snrm2, float, float);
IMPL(rocblas_dnrm2, double, double);
IMPL(rocblas_scnrm2, rocblas_float_complex, float);
IMPL(rocblas_dznrm2, rocblas_double_complex, double);

#undef IMPL

} // extern "C"
