/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "logging.h"
#include "trtri_host.hpp"
#include "utility.h"


template <typename>
constexpr char rocblas_trtri_name[] = "unknown";
template <>
constexpr char rocblas_trtri_name<float>[] = "rocblas_strtri_strided_batched";
template <>
constexpr char rocblas_trtri_name<double>[] = "rocblas_dtrtri_strided_batched";

template <rocblas_int NB, typename T>
rocblas_status rocblas_trtri_strided_batched_impl(rocblas_handle   handle,
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
        status = rocblas_trtri_small_strided_batched<NB>(
            handle, uplo, diag, n, A, lda, bsa, 0, invA, ldinvA, bsinvA, 0, batch_count, 1);
    }
    else
    {
        // Compute the optimal size for temporary device memory
        size_t size = rocblas_trtri_strided_batched_temp_size<NB>(n, batch_count) * sizeof(T);

        // If size is queried, set optimal size
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(size);

        // Allocate memory
        auto C_tmp = handle->device_malloc(size);
        if(!C_tmp)
            return rocblas_status_memory_error;

        status = rocblas_trtri_large_strided_batched<NB>(handle,
                                                            uplo,
                                                            diag,
                                                            n,
                                                            A,
                                                            lda,
                                                            bsa,
                                                            0,
                                                            invA,
                                                            ldinvA,
                                                            bsinvA,
                                                            0,
                                                            batch_count,
                                                            1,
                                                            (T*)C_tmp);
    }

    return status;
}
