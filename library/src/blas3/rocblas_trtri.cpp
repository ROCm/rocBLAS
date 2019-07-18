/* ************************************************************************
 *  * Copyright 2016-2019 Advanced Micro Devices, Inc.
 *  *
 *  * ************************************************************************ */
#include "trtri_trsm.hpp"

namespace
{
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

        // TODO: Add logging

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

        size_t size = rocblas_trtri_batched_temp_size<NB>(n, 1) * sizeof(T);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(size);

        auto mem = handle->device_malloc(size);
        if(!mem)
            return rocblas_status_memory_error;

        return rocblas_trtri_template<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA, (T*)mem);
    }

} // namespace

/* ============================================================================================ */

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
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

rocblas_status rocblas_dtrtri(rocblas_handle   handle,
                              rocblas_fill     uplo,
                              rocblas_diagonal diag,
                              rocblas_int      n,
                              const double*    A,
                              rocblas_int      lda,
                              double*          invA,
                              rocblas_int      ldinvA)
{
    constexpr rocblas_int NB = 16;
    return rocblas_trtri_impl<NB>(handle, uplo, diag, n, A, lda, invA, ldinvA);
}

} // extern "C"
