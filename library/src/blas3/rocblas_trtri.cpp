/* ************************************************************************
 *  * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *  *
 *  * ************************************************************************ */
#include "rocblas_trtri.hpp"
#include "logging.h"
#include "utility.h"

namespace
{
    template <typename>
    constexpr char rocblas_trtri_name[] = "unknown";
    template <>
    constexpr char rocblas_trtri_name<float>[] = "rocblas_strtri";
    template <>
    constexpr char rocblas_trtri_name<double>[] = "rocblas_dtrtri";
    template <>
    constexpr char rocblas_trtri_name<rocblas_float_complex>[] = "rocblas_ctrtri";
    template <>
    constexpr char rocblas_trtri_name<rocblas_double_complex>[] = "rocblas_ztrtri";

    template <rocblas_int NB, typename T>
    rocblas_status rocblas_trtri_impl(rocblas_handle   handle,
                                      rocblas_fill     uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int      n,
                                      const T*         A,
                                      rocblas_int      lda,
                                      T*               invA,
                                      rocblas_int      ldinvA)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        size_t size = rocblas_trtri_temp_size<NB>(n, 1) * sizeof(T);
        if(handle->is_device_memory_size_query())
        {
            if(!n)
                return rocblas_status_size_unchanged;
            return handle->set_optimal_device_memory_size(size);
        }

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_trtri_name<T>, uplo, diag, n, A, lda, invA, ldinvA);

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_trtri_name<T>,
                        "uplo",
                        rocblas_fill_letter(uplo),
                        "diag",
                        rocblas_diag_letter(diag),
                        "N",
                        n,
                        "lda",
                        lda,
                        "ldinvA",
                        ldinvA);

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_invalid_value;
        if(n < 0 || lda < n)
            return rocblas_status_invalid_size;
        if(!n)
            return rocblas_status_success;
        if(!A || !invA)
            return rocblas_status_invalid_pointer;

        auto mem = handle->device_malloc(size);
        if(!mem)
            return rocblas_status_memory_error;

        return rocblas_trtri_template<NB, false, false, T>(handle,
                                                           uplo,
                                                           diag,
                                                           n,
                                                           A,
                                                           0,
                                                           lda,
                                                           lda * n,
                                                           0,
                                                           invA,
                                                           0,
                                                           ldinvA,
                                                           ldinvA * n,
                                                           0,
                                                           1,
                                                           1,
                                                           (T*)mem);
    }

}

/*
 * ===========================================================================
 *    C interface
 * ===========================================================================
 */

extern "C" {
rocblas_status rocblas_strtri(rocblas_handle   handle,
                              rocblas_fill     uplo,
                              rocblas_diagonal diag,
                              rocblas_int      n,
                              const float*     A,
                              rocblas_int      lda,
                              float*           invA,
                              rocblas_int      ldinvA)
try
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtrtri(rocblas_handle   handle,
                              rocblas_fill     uplo,
                              rocblas_diagonal diag,
                              rocblas_int      n,
                              const double*    A,
                              rocblas_int      lda,
                              double*          invA,
                              rocblas_int      ldinvA)
try
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctrtri(rocblas_handle               handle,
                              rocblas_fill                 uplo,
                              rocblas_diagonal             diag,
                              rocblas_int                  n,
                              const rocblas_float_complex* A,
                              rocblas_int                  lda,
                              rocblas_float_complex*       invA,
                              rocblas_int                  ldinvA)
try
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztrtri(rocblas_handle                handle,
                              rocblas_fill                  uplo,
                              rocblas_diagonal              diag,
                              rocblas_int                   n,
                              const rocblas_double_complex* A,
                              rocblas_int                   lda,
                              rocblas_double_complex*       invA,
                              rocblas_int                   ldinvA)
try
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
