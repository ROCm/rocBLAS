/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#include "logging.h"
#include "rocblas_trtri.hpp"
#include "utility.h"

namespace
{

    template <typename>
    constexpr char rocblas_trtri_name[] = "unknown";
    template <>
    constexpr char rocblas_trtri_name<float>[] = "rocblas_strtri_batched";
    template <>
    constexpr char rocblas_trtri_name<double>[] = "rocblas_dtrtri_batched";

    template <rocblas_int NB, typename T>
    rocblas_status rocblas_trtri_batched_impl(rocblas_handle   handle,
                                              rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const T* const   A[],
                                              rocblas_int      lda,
                                              T*               invA[],
                                              rocblas_int      ldinvA,
                                              rocblas_int      batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(
                handle, rocblas_trtri_name<T>, uplo, diag, n, A, lda, invA, ldinvA, batch_count);

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
                        ldinvA,
                        "batch_count",
                        batch_count);

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(n < 0)
            return rocblas_status_invalid_size;
        if(!A)
            return rocblas_status_invalid_pointer;
        if(lda < n)
            return rocblas_status_invalid_size;
        if(!invA)
            return rocblas_status_invalid_pointer;
        if(ldinvA < n || batch_count < 0)
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
            status = rocblas_trtri_small<NB, T>(
                handle, uplo, diag, n, A, 0, lda, 0, 0, invA, 0, ldinvA, 0, 0, batch_count, 1);
        }
        else
        {
            // Compute the optimal size for temporary device memory
            size_t els   = rocblas_trtri_temp_size<NB>(n, 1);
            size_t size  = els * batch_count * sizeof(T);
            size_t sizep = batch_count * sizeof(T*);

            // If size is queried, set optimal size
            if(handle->is_device_memory_size_query())
                return handle->set_optimal_device_memory_size(size, sizep);

            // Allocate memory
            auto mem = handle->device_malloc(size, sizep);
            if(!mem)
            {
                return rocblas_status_memory_error;
            }
            void* C_tmp;
            void* C_tmp_arr;
            std::tie(C_tmp, C_tmp_arr) = mem;

            T* C_tmp_host[batch_count];
            for(int b = 0; b < batch_count; b++)
            {
                C_tmp_host[b] = (T*)C_tmp + b * els;
            }
            RETURN_IF_HIP_ERROR(
                hipMemcpy(C_tmp_arr, C_tmp_host, batch_count * sizeof(T*), hipMemcpyHostToDevice));

            status = rocblas_trtri_large<NB, true, false, T>(handle,
                                                             uplo,
                                                             diag,
                                                             n,
                                                             A,
                                                             0,
                                                             lda,
                                                             0,
                                                             0,
                                                             invA,
                                                             0,
                                                             ldinvA,
                                                             0,
                                                             0,
                                                             batch_count,
                                                             1,
                                                             (T**)C_tmp_arr);
        }

        return status;
    }

}

extern "C" {
rocblas_status rocblas_strtri_batched(rocblas_handle     handle,
                                      rocblas_fill       uplo,
                                      rocblas_diagonal   diag,
                                      rocblas_int        n,
                                      const float* const A[],
                                      rocblas_int        lda,
                                      float*             invA[],
                                      rocblas_int        ldinvA,
                                      rocblas_int        batch_count)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_batched_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}

rocblas_status rocblas_dtrtri_batched(rocblas_handle      handle,
                                      rocblas_fill        uplo,
                                      rocblas_diagonal    diag,
                                      rocblas_int         n,
                                      const double* const A[],
                                      rocblas_int         lda,
                                      double*             invA[],
                                      rocblas_int         ldinvA,
                                      rocblas_int         batch_count)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_batched_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}

} // extern "C"
