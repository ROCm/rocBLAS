/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "../blas3/trtri_trsm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "rocblas_gemv.hpp"
#include "utility.h"
#include <algorithm>
#include <cstdio>
#include <tuple>

namespace
{
    using std::max;
    using std::min;

    constexpr rocblas_int NB_X        = 1024;
    constexpr rocblas_int STRSV_BLOCK = 128;
    constexpr rocblas_int DTRSV_BLOCK = 128;

    template <typename T>
    constexpr T negative_one = -1;
    template <typename T>
    constexpr T zero = 0;
    template <typename T>
    constexpr T one = 1;

    template <typename T>
    __global__ void flip_vector_kernel(T* __restrict__ data,
                                       T* __restrict__ data_end,
                                       rocblas_int size,
                                       rocblas_int abs_incx)
    {
        ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if(tx < size)
        {
            auto offset       = tx * abs_incx;
            auto temp         = data[offset];
            data[offset]      = data_end[-offset];
            data_end[-offset] = temp;
        }
    }

    template <typename T>
    void flip_vector(rocblas_handle handle, T* data, rocblas_int m, rocblas_int abs_incx)
    {
        T*          data_end = data + (m - 1) * abs_incx;
        rocblas_int size     = (m + 1) / 2;
        rocblas_int blocksX  = (size - 1) / NB_X + 1;
        dim3        grid     = blocksX;
        dim3        threads  = NB_X;

        hipLaunchKernelGGL(flip_vector_kernel,
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           data,
                           data_end,
                           size,
                           abs_incx);
    }

    template <typename T>
    __global__ void strided_vector_copy_kernel(T* __restrict__ dst,
                                               rocblas_int dst_incx,
                                               const T* __restrict__ src,
                                               rocblas_int src_incx,
                                               rocblas_int size)
    {
        ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if(tx < size)
            dst[tx * dst_incx] = src[tx * src_incx];
    }

    template <typename T>
    void strided_vector_copy(rocblas_handle handle,
                             T*             dst,
                             rocblas_int    dst_incx,
                             T*             src,
                             rocblas_int    src_incx,
                             rocblas_int    size)
    {
        rocblas_int blocksX = (size - 1) / NB_X + 1;
        dim3        grid    = blocksX;
        dim3        threads = NB_X;

        hipLaunchKernelGGL(strided_vector_copy_kernel,
                           grid,
                           threads,
                           0,
                           handle->rocblas_stream,
                           dst,
                           dst_incx,
                           src,
                           src_incx,
                           size);
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsv_left(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_int       m,
                                     const T*          A,
                                     rocblas_int       lda,
                                     T*                B,
                                     rocblas_int       incx,
                                     const T*          invA,
                                     T*                X)
    {
        rocblas_int i, jb;

        if(transA == rocblas_operation_none)
        {
            if(uplo == rocblas_fill_lower)
            {
                // left, lower no-transpose
                jb = min(BLOCK, m);
                rocblas_gemv_template<T>(handle,
                                         transA,
                                         jb,
                                         jb,
                                         &one<T>,
                                         invA,
                                         0,
                                         BLOCK,
                                         0,
                                         B,
                                         0,
                                         incx,
                                         0,
                                         &zero<T>,
                                         X,
                                         0,
                                         1,
                                         0,
                                         1);

                if(BLOCK < m)
                {
                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             m - BLOCK,
                                             BLOCK,
                                             &negative_one<T>,
                                             A + BLOCK,
                                             0,
                                             lda,
                                             0,
                                             X,
                                             0,
                                             1,
                                             0,
                                             &one<T>,
                                             B + BLOCK * incx,
                                             0,
                                             incx,
                                             0,
                                             1);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);

                        rocblas_gemv_template<T>(handle,
                                                 transA,
                                                 jb,
                                                 jb,
                                                 &one<T>,
                                                 invA + i * BLOCK,
                                                 0,
                                                 BLOCK,
                                                 0,
                                                 B + i * incx,
                                                 0,
                                                 incx,
                                                 0,
                                                 &zero<T>,
                                                 X + i,
                                                 0,
                                                 1,
                                                 0,
                                                 1);
                        if(i + BLOCK < m)
                            rocblas_gemv_template<T>(handle,
                                                     transA,
                                                     m - i - BLOCK,
                                                     BLOCK,
                                                     &negative_one<T>,
                                                     A + i + BLOCK + i * lda,
                                                     0,
                                                     lda,
                                                     0,
                                                     X + i,
                                                     0,
                                                     1,
                                                     0,
                                                     &one<T>,
                                                     B + (i + BLOCK) * incx,
                                                     0,
                                                     incx,
                                                     0,
                                                     1);
                    }
                }
            }
            else
            {
                // left, upper no-transpose
                jb = m % BLOCK == 0 ? BLOCK : m % BLOCK;
                i  = m - jb;

                // if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35}
                rocblas_gemv_template<T>(handle,
                                         transA,
                                         jb,
                                         jb,
                                         &one<T>,
                                         invA + i * BLOCK,
                                         0,
                                         BLOCK,
                                         0,
                                         B + i * incx,
                                         0,
                                         incx,
                                         0,
                                         &zero<T>,
                                         X + i,
                                         0,
                                         1,
                                         0,
                                         1);

                if(i >= BLOCK)
                {
                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             i,
                                             jb,
                                             &negative_one<T>,
                                             A + i * lda,
                                             0,
                                             lda,
                                             0,
                                             X + i,
                                             0,
                                             1,
                                             0,
                                             &one<T>,
                                             B,
                                             0,
                                             incx,
                                             0,
                                             1);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        //{32, 35, 32, 32, 35, 35}
                        rocblas_gemv_template<T>(handle,
                                                 transA,
                                                 BLOCK,
                                                 BLOCK,
                                                 &one<T>,
                                                 invA + i * BLOCK,
                                                 0,
                                                 BLOCK,
                                                 0,
                                                 B + i * incx,
                                                 0,
                                                 incx,
                                                 0,
                                                 &zero<T>,
                                                 X + i,
                                                 0,
                                                 1,
                                                 0,
                                                 1);

                        if(i >= BLOCK)
                            rocblas_gemv_template<T>(handle,
                                                     transA,
                                                     i,
                                                     BLOCK,
                                                     &negative_one<T>,
                                                     A + i * lda,
                                                     0,
                                                     lda,
                                                     0,
                                                     X + i,
                                                     0,
                                                     1,
                                                     0,
                                                     &one<T>,
                                                     B,
                                                     0,
                                                     incx,
                                                     0,
                                                     1);
                    }
                }
            }
        }
        else
        { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
            if(uplo == rocblas_fill_lower)
            {
                // left, lower transpose
                jb = m % BLOCK == 0 ? BLOCK : m % BLOCK;
                i  = m - jb;

                rocblas_gemv_template<T>(handle,
                                         transA,
                                         jb,
                                         jb,
                                         &one<T>,
                                         invA + i * BLOCK,
                                         0,
                                         BLOCK,
                                         0,
                                         B + i * incx,
                                         0,
                                         incx,
                                         0,
                                         &zero<T>,
                                         X + i,
                                         0,
                                         1,
                                         0,
                                         1);

                if(i - BLOCK >= 0)
                {
                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             jb,
                                             i,
                                             &negative_one<T>,
                                             A + i,
                                             0,
                                             lda,
                                             0,
                                             X + i,
                                             0,
                                             1,
                                             0,
                                             &one<T>,
                                             B,
                                             0,
                                             incx,
                                             0,
                                             1);

                    // remaining blocks
                    for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                    {
                        rocblas_gemv_template<T>(handle,
                                                 transA,
                                                 BLOCK,
                                                 BLOCK,
                                                 &one<T>,
                                                 invA + i * BLOCK,
                                                 0,
                                                 BLOCK,
                                                 0,
                                                 B + i * incx,
                                                 0,
                                                 incx,
                                                 0,
                                                 &zero<T>,
                                                 X + i,
                                                 0,
                                                 1,
                                                 0,
                                                 1);

                        if(i >= BLOCK)
                            rocblas_gemv_template<T>(handle,
                                                     transA,
                                                     BLOCK,
                                                     i,
                                                     &negative_one<T>,
                                                     A + i,
                                                     0,
                                                     lda,
                                                     0,
                                                     X + i,
                                                     0,
                                                     1,
                                                     0,
                                                     &one<T>,
                                                     B,
                                                     0,
                                                     incx,
                                                     0,
                                                     1);
                    }
                }
            }
            else
            {
                // left, upper transpose
                jb = min(BLOCK, m);
                rocblas_gemv_template<T>(handle,
                                         transA,
                                         jb,
                                         jb,
                                         &one<T>,
                                         invA,
                                         0,
                                         BLOCK,
                                         0,
                                         B,
                                         0,
                                         incx,
                                         0,
                                         &zero<T>,
                                         X,
                                         0,
                                         1,
                                         0,
                                         1);

                if(BLOCK < m)
                {
                    rocblas_gemv_template<T>(handle,
                                             transA,
                                             BLOCK,
                                             m - BLOCK,
                                             &negative_one<T>,
                                             A + BLOCK * lda,
                                             0,
                                             lda,
                                             0,
                                             X,
                                             0,
                                             1,
                                             0,
                                             &one<T>,
                                             B + BLOCK * incx,
                                             0,
                                             incx,
                                             0,
                                             1);

                    // remaining blocks
                    for(i = BLOCK; i < m; i += BLOCK)
                    {
                        jb = min(m - i, BLOCK);
                        rocblas_gemv_template<T>(handle,
                                                 transA,
                                                 jb,
                                                 jb,
                                                 &one<T>,
                                                 invA + i * BLOCK,
                                                 0,
                                                 BLOCK,
                                                 0,
                                                 B + i * incx,
                                                 0,
                                                 incx,
                                                 0,
                                                 &zero<T>,
                                                 X + i,
                                                 0,
                                                 1,
                                                 0,
                                                 1);

                        if(i + BLOCK < m)
                            rocblas_gemv_template<T>(handle,
                                                     transA,
                                                     BLOCK,
                                                     m - i - BLOCK,
                                                     &negative_one<T>,
                                                     A + i + (i + BLOCK) * lda,
                                                     0,
                                                     lda,
                                                     0,
                                                     X + i,
                                                     0,
                                                     1,
                                                     0,
                                                     &one<T>,
                                                     B + (i + BLOCK) * incx,
                                                     0,
                                                     incx,
                                                     0,
                                                     1);
                    }
                }
            }
        } // transpose

        return rocblas_status_success;
    }

    template <rocblas_int BLOCK, typename T>
    rocblas_status special_trsv_template(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         rocblas_int       m,
                                         const T*          A,
                                         rocblas_int       lda,
                                         T*                B,
                                         ptrdiff_t         incx,
                                         const T*          invA,
                                         T*                x_temp)
    {
        bool   parity = (transA == rocblas_operation_none) ^ (uplo == rocblas_fill_lower);
        size_t R      = m / BLOCK;

        for(size_t r = 0; r < R; r++)
        {
            size_t q = R - r;
            size_t j = parity ? q - 1 : r;

            // copy a BLOCK*n piece we are solving at a time
            strided_vector_copy<T>(handle, x_temp, 1, B + incx * j * BLOCK, incx, BLOCK);

            if(r)
            {
                rocblas_int M = BLOCK;
                rocblas_int N = BLOCK;
                const T*    A_current;
                T*          B_current = parity ? B + q * BLOCK * incx : B;

                if(transA == rocblas_operation_none)
                {
                    N *= r;
                    A_current = parity ? A + BLOCK * ((lda + 1) * q - 1) : A + N;
                }
                else
                {
                    M *= r;
                    A_current = parity ? A + BLOCK * ((lda + 1) * q - lda) : A + M * lda;
                }

                rocblas_gemv_template<T>(handle,
                                         transA,
                                         M,
                                         N,
                                         &negative_one<T>,
                                         A_current,
                                         0,
                                         lda,
                                         0,
                                         B_current,
                                         0,
                                         incx,
                                         0,
                                         &one<T>,
                                         x_temp,
                                         0,
                                         1,
                                         0,
                                         1);
            }

            rocblas_gemv_template<T>(handle,
                                     transA,
                                     BLOCK,
                                     BLOCK,
                                     &one<T>,
                                     invA + j * BLOCK * BLOCK,
                                     0,
                                     BLOCK,
                                     0,
                                     x_temp,
                                     0,
                                     1,
                                     0,
                                     &zero<T>,
                                     B + j * BLOCK * incx,
                                     0,
                                     incx,
                                     0,
                                     1);
        }

        return rocblas_status_success;
    }

    template <typename>
    constexpr char rocblas_trsv_name[] = "unknown";
    template <>
    constexpr char rocblas_trsv_name<float>[] = "rocblas_strsv";
    template <>
    constexpr char rocblas_trsv_name<double>[] = "rocblas_dtrsv";

    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsv_ex_impl(rocblas_handle    handle,
                                        rocblas_fill      uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal  diag,
                                        rocblas_int       m,
                                        const T*          A,
                                        rocblas_int       lda,
                                        T*                B,
                                        rocblas_int       incx,
                                        const T*          supplied_invA      = nullptr,
                                        rocblas_int       supplied_invA_size = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto layer_mode = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_trsv_name<T>, uplo, transA, diag, m, A, lda, B, incx);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if(handle->pointer_mode == rocblas_pointer_mode_host)
                    log_bench(handle,
                              "./rocblas-bench -f trsv -r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-m",
                              m,
                              "--lda",
                              lda,
                              "--incx",
                              incx);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_trsv_name<T>,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "diag",
                            diag_letter,
                            "M",
                            m,
                            "lda",
                            lda,
                            "incx",
                            incx);
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(!A || !B)
            return rocblas_status_invalid_pointer;
        if(m < 0 || lda < m || lda < 1 || !incx)
            return rocblas_status_invalid_size;

        // quick return if possible.
        // return rocblas_status_size_unchanged if device memory size query
        if(!m)
            return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                         : rocblas_status_success;

        // Whether size is an exact multiple of blocksize
        const bool exact_blocks = (m % BLOCK) == 0;

        // perf_status indicates whether optimal performance is obtainable with available memory
        rocblas_status perf_status = rocblas_status_success;

        size_t invA_bytes   = 0;
        size_t c_temp_bytes = 0;

        // For user-supplied invA, check to make sure size is large enough
        // If not large enough, indicate degraded performance and ignore supplied invA
        if(supplied_invA && supplied_invA_size / BLOCK < m)
        {
            static int msg = fputs("WARNING: TRSV invA_size argument is too small; invA argument "
                                   "is being ignored; TRSV performance is degraded\n",
                                   stderr);
            perf_status    = rocblas_status_perf_degraded;
            supplied_invA  = nullptr;
        }

        if(!supplied_invA)
        {
            // Only allocate bytes for invA if supplied_invA == nullptr or supplied_invA_size is too small
            invA_bytes = sizeof(T) * BLOCK * m;

            // When m < BLOCK, C is unnecessary for trtri
            c_temp_bytes = (m / BLOCK) * (sizeof(T) * (BLOCK / 2) * (BLOCK / 2));

            // For the TRTRI last diagonal block we need remainder space if m % BLOCK != 0
            if(!exact_blocks)
            {
                // TODO: Make this more accurate -- right now it's much larger than necessary
                size_t remainder_bytes = sizeof(T) * ROCBLAS_TRTRI_NB * BLOCK * 2;

                // C is the maximum of the temporary space needed for TRTRI
                c_temp_bytes = max(c_temp_bytes, remainder_bytes);
            }
        }

        // Temporary solution vector
        // If the special solver can be used, then only BLOCK words are needed instead of m words
        size_t x_temp_bytes = exact_blocks ? sizeof(T) * BLOCK : sizeof(T) * m;

        // X and C temporaries can share space, so the maximum size is allocated
        size_t x_c_temp_bytes = max(x_temp_bytes, c_temp_bytes);

        // If this is a device memory size query, set optimal size and return changed status
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(x_c_temp_bytes, invA_bytes);

        // Attempt to allocate optimal memory size, returning error if failure
        auto mem = handle->device_malloc(x_c_temp_bytes, invA_bytes);
        if(!mem)
            return rocblas_status_memory_error;

        // Get pointers to allocated device memory
        // Note: Order of pointers in std::tie(...) must match order of sizes in handle->device_malloc(...)
        void* x_temp;
        void* invA;
        std::tie(x_temp, invA) = mem;

        // Temporarily switch to host pointer mode, restoring on return
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        rocblas_status status = rocblas_status_success;
        if(supplied_invA)
            invA = const_cast<T*>(supplied_invA);
        else
        {
            // batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
            auto c_temp = x_temp; // Uses same memory as x_temp
            status      = rocblas_trtri_trsm_template<BLOCK>(
                handle, (T*)c_temp, uplo, diag, m, A, lda, (T*)invA);
            if(status != rocblas_status_success)
                return status;
        }

        if(transA == rocblas_operation_conjugate_transpose)
            transA = rocblas_operation_transpose;

        // TODO: workaround to fix negative incx issue
        rocblas_int abs_incx = incx < 0 ? -incx : incx;
        if(incx < 0)
            flip_vector(handle, B, m, abs_incx);

        if(exact_blocks)
        {
            status = special_trsv_template<BLOCK>(
                handle, uplo, transA, diag, m, A, lda, B, abs_incx, (const T*)invA, (T*)x_temp);
            if(status != rocblas_status_success)
                return status;

            // TODO: workaround to fix negative incx issue
            if(incx < 0)
                flip_vector(handle, B, m, abs_incx);
        }
        else
        {
            status = rocblas_trsv_left<BLOCK>(
                handle, uplo, transA, m, A, lda, B, abs_incx, (const T*)invA, (T*)x_temp);
            if(status != rocblas_status_success)
                return status;

            // copy solution X into B
            // TODO: workaround to fix negative incx issue
            strided_vector_copy(handle,
                                B,
                                abs_incx,
                                incx < 0 ? (T*)x_temp + m - 1 : (T*)x_temp,
                                incx < 0 ? -1 : 1,
                                m);
        }
        return perf_status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_strsv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             const float*      A,
                             rocblas_int       lda,
                             float*            x,
                             rocblas_int       incx)
{
    return rocblas_trsv_ex_impl<STRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

rocblas_status rocblas_dtrsv(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transA,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             const double*     A,
                             rocblas_int       lda,
                             double*           x,
                             rocblas_int       incx)
{
    return rocblas_trsv_ex_impl<DTRSV_BLOCK>(handle, uplo, transA, diag, m, A, lda, x, incx);
}

rocblas_status rocblas_trsv_ex(rocblas_handle    handle,
                               rocblas_fill      uplo,
                               rocblas_operation transA,
                               rocblas_diagonal  diag,
                               rocblas_int       m,
                               const void*       A,
                               rocblas_int       lda,
                               void*             x,
                               rocblas_int       incx,
                               const void*       invA,
                               rocblas_int       invA_size,
                               rocblas_datatype  compute_type)

{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsv_ex_impl<DTRSV_BLOCK>(handle,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 m,
                                                 static_cast<const double*>(A),
                                                 lda,
                                                 static_cast<double*>(x),
                                                 incx,
                                                 static_cast<const double*>(invA),
                                                 invA_size);

    case rocblas_datatype_f32_r:
        return rocblas_trsv_ex_impl<STRSV_BLOCK>(handle,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 m,
                                                 static_cast<const float*>(A),
                                                 lda,
                                                 static_cast<float*>(x),
                                                 incx,
                                                 static_cast<const float*>(invA),
                                                 invA_size);

    default:
        return rocblas_status_not_implemented;
    }
}

} // extern "C"
