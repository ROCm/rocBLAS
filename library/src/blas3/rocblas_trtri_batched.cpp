/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas_trtri_batched.hpp"
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

    template <rocblas_int NB, typename T>
    rocblas_status rocblas_trtri_batched_impl(rocblas_handle   handle,
                                              rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const T*         A,
                                              rocblas_int      lda,
                                              rocblas_int      bsa,
                                              T*               invA,
                                              rocblas_int      ldinvA,
                                              rocblas_int      bsinvA,
                                              rocblas_int      batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle,
                      rocblas_trtri_name<T>,
                      uplo,
                      diag,
                      n,
                      A,
                      lda,
                      bsa,
                      invA,
                      ldinvA,
                      bsinvA,
                      batch_count);

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
                        "bsa",
                        bsa,
                        "ldinvA",
                        ldinvA,
                        "bsinvA",
                        bsinvA,
                        "batch_count",
                        batch_count);

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(n < 0)
            return rocblas_status_invalid_size;
        if(!A)
            return rocblas_status_invalid_pointer;
        if(lda < n || bsa < lda * n)
            return rocblas_status_invalid_size;
        if(!invA)
            return rocblas_status_invalid_pointer;
        if(ldinvA < n || bsinvA < ldinvA * n || batch_count < 0)
            return rocblas_status_invalid_size;

        // For small n or zero batch_count, and device memory size query, return size unchanged
        if((n <= NB || !batch_count) && handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;

        // Quick return if possible.
        if(!n || !batch_count)
            return rocblas_status_success;

        rocblas_status status;
        if(n <= NB)
        {
            status = rocblas_trtri_small_batched<NB>(
                handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
        }
        else
        {
            // Compute the optimal size for temporary device memory
            size_t size = rocblas_trtri_batched_temp_size<NB>(n, batch_count) * sizeof(T);

            // If size is queried, set optimal size
            if(handle->is_device_memory_size_query())
                return handle->set_optimal_device_memory_size(size);

            // Allocate memory
            auto C_tmp = handle->device_malloc(size);
            if(!C_tmp)
                return rocblas_status_memory_error;

            status = rocblas_trtri_large_batched<NB>(
                handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count, (T*)C_tmp);
        }

        return status;
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C interface
 *    This function is called by trsm
 * ===========================================================================
 */

extern "C" {
rocblas_status rocblas_strtri_batched(rocblas_handle   handle,
                                      rocblas_fill     uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int      n,
                                      const float*     A,
                                      rocblas_int      lda,
                                      rocblas_int      bsa,
                                      float*           invA,
                                      rocblas_int      ldinvA,
                                      rocblas_int      bsinvA,
                                      rocblas_int      batch_count)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_batched_impl<NB>(
        handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}

rocblas_status rocblas_dtrtri_batched(rocblas_handle   handle,
                                      rocblas_fill     uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int      n,
                                      const double*    A,
                                      rocblas_int      lda,
                                      rocblas_int      bsa,
                                      double*          invA,
                                      rocblas_int      ldinvA,
                                      rocblas_int      bsinvA,
                                      rocblas_int      batch_count)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_batched_impl<NB>(
        handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
}

} // extern "C"
