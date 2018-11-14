/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "rocblas_trsm.hpp"

#include "status.h"
#include "definitions.h"
#include "trsv_device.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"
#include "rocblas_unique_ptr.hpp"

template <typename T, const rocblas_int NB_X, const rocblas_int NB_Y>
__global__ void trsv_kernel_device_pointer(rocblas_operation transA,
                                            rocblas_int m,
                                            const T* __restrict__ A,
                                            rocblas_int lda,
                                            T* __restrict__ x,
                                            rocblas_int incx)
{
    trsv_device<T, 128,NB_X, NB_Y>(m, A, lda, x, incx);
}

template <typename T>
__global__ void strided_vector_copy_to_temp(T* x,
                                          rocblas_int m,
                                          T* x_temp,
                                          rocblas_int incx)
{
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    int id_x = ty*hipBlockDim_x*hipGridDim_x+tx;
    int id_temp = id_x/incx;

    if(id_temp<m && id_x%incx == 0)
        x_temp[id_temp] = x[id_x];

}

template <typename T>
__global__ void strided_vector_copy_from_temp(T* x,
                                            rocblas_int m,
                                            T* x_temp,
                                            rocblas_int incx)
{
    size_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    int id_x = ty*hipBlockDim_x*hipGridDim_x+tx;
    int id_temp = id_x/incx;

    if(id_temp<m && id_x%incx == 0)
        x[id_x] = x_temp[id_temp];

}


template <typename T>
void strided_vector_copy(hipStream_t rocblas_stream,
                        T* x,
                        rocblas_int m,
                        T* x_temp,
                        rocblas_int incx,
                        bool to_temp)
{
    rocblas_int blocksX = ((m - 1) / 128) + 1; // parameters for device kernel
    rocblas_int blocksY = ((m - 1) / 8) + 1;
    dim3 grid(blocksX, blocksY, 1);
    dim3 threads(128, 8, 1);

    if(to_temp)
        hipLaunchKernelGGL(strided_vector_copy_to_temp,
                        dim3(grid),
                        dim3(threads),
                        0,
                        rocblas_stream,
                        x,
                        m,
                        x_temp,
                        incx);
    else
        hipLaunchKernelGGL(strided_vector_copy_from_temp,
                        dim3(grid),
                        dim3(threads),
                        0,
                        rocblas_stream,
                        x,
                        m,
                        x_temp,
                        incx);
}

/*! \brief BLAS Level 2 API

    \details
    trsv solves

         A*x = b or A^T*x = b,

    where x and b are vectors and A is a triangular matrix.

    The vector x is overwritten on b.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.

    @param[in]
    uplo    rocblas_fill.
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     rocblas_operation

    @param[in]
    diag    rocblas_diagonal.
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         rocblas_int
              m specifies the number of rows of b. m >= 0.

    @param[in]
    A         pointer storing matrix A on the GPU,
              of dimension ( lda, m )

    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
              lda >= max( 1, m ).

    @param[in]
    x         pointer storing vector x on the GPU.

    @param[in]
    incx      specifies the increment for the elements of x.

    ********************************************************************/

template <typename T, rocblas_int BLOCK>
rocblas_status rocblas_trsv_template(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     const T* A,
                                     rocblas_int lda,
                                     T* x,
                                     rocblas_int incx)
{
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xtrsv"),
                  uplo,
                  transA,
                  diag,
                  m,
                  (const void*&)A,
                  lda,
                  (const void*&)x,
                  incx);

        std::string uplo_letter   = rocblas_fill_letter(uplo);
        std::string transA_letter = rocblas_transpose_letter(transA);
        std::string diag_letter   = rocblas_diag_letter(diag);

        log_bench(handle,
                  "./rocblas-bench -f trsv -r",
                  replaceX<T>("X"),
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
    else
    {
        log_trace(handle,
                  replaceX<T>("rocblas_Xtrsv"),
                  uplo,
                  transA,
                  diag,
                  m,
                  (const void*&)A,
                  lda,
                  (const void*&)x,
                  incx);
    }

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_not_implemented;
    else if(nullptr == A)
        return rocblas_status_invalid_pointer;
    else if(nullptr == x)
        return rocblas_status_invalid_pointer;
    else if(m < 0)
        return rocblas_status_invalid_size;
    else if(lda < m || lda < 1)
        return rocblas_status_invalid_size;
    else if(0 == incx)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(m == 0 )
        return rocblas_status_success;



    //Calling TRSM for now 

    rocblas_int size_x = ( 1 + ( m - 1 )*abs( incx ) ); //  at least according to netlib... wrong?
    std::vector<T> hx_orig(size_x);
    std::vector<T> hx_mod(m);
    auto      dx_managed = rocblas_unique_ptr{rocblas::device_malloc(sizeof(T) * m),
                                            rocblas::device_free};
    T*        dx_mod   = (T*)dx_managed.get();
    T alpha_h        = 1.0f;
    auto alpha_d_managed =
        rocblas_unique_ptr{rocblas::device_malloc(sizeof(T)), rocblas::device_free};
    T* alpha_d = (T*)alpha_d_managed.get();
    hipMemcpy(alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice);
    rocblas_status status;

    rocblas_pointer_mode pointer_mode = handle->pointer_mode;

    if(incx>1)
    {
        strided_vector_copy<T>(handle->rocblas_stream,
                            x,
                            m,
                            dx_mod,
                            incx,
                            true);

        rocblas_trsm_template<T, BLOCK>(handle,
                                        rocblas_side_left,
                                        uplo,
                                        transA,
                                        diag,
                                        m,
                                        1,
                                        pointer_mode==rocblas_pointer_mode_host ? &alpha_h:alpha_d,
                                        A,
                                        lda,
                                        dx_mod,
                                        m);

        strided_vector_copy<T>(handle->rocblas_stream,
                            x,
                            m,
                            dx_mod,
                            incx,
                            false);
    }
    else
    {
        status = rocblas_trsm_template<T, BLOCK>(handle,
                                                rocblas_side_left,
                                                uplo,
                                                transA,
                                                diag,
                                                m,
                                                1,
                                                pointer_mode==rocblas_pointer_mode_host ? &alpha_h:alpha_d,
                                                A,
                                                lda,
                                                x,
                                                m);
    }

    return status;

    // //copy original x to host mem
    // hipMemcpy(hx_orig.data(), x, sizeof(T) * size_x, hipMemcpyDeviceToHost);

    // //take the useful values out of x
    // for(int i =0; i<m; i++)
    //     hx_mod[i] = hx_orig[i*incx];

    // //save useful values back in device to use in trsm
    // hipMemcpy(dx_mod, hx_mod.data(), sizeof(T) * m, hipMemcpyHostToDevice);

    // rocblas_status status = rocblas_trsm_template<T, BLOCK>(handle,
    //                                                         rocblas_side_left,
    //                                                         uplo,
    //                                                         transA,
    //                                                         diag,
    //                                                         m,
    //                                                         1,
    //                                                         A,
    //                                                         lda,
    //                                                         dx_mod,
    //                                                         m);
    // //copy x results in to host mem
    // hipMemcpy(hx_mod.data(), dx_mod, sizeof(T) * m, hipMemcpyDeviceToHost);

    // //replace modified values in the full array
    // for(int i =0; i<m; i++)
    //     hx_orig[i*incx] = hx_mod[i];
    
    // //make changes on original device memory sent in by function argument
    // hipMemcpy(x, hx_orig.data(), sizeof(T) * size_x, hipMemcpyHostToDevice);

    // if(std::is_same<T, float>::value)
    //     return rocblas_strsm(   handle,
    //                             rocblas_side_left,
    //                             uplo,
    //                             transA,
    //                             diag,
    //                             m,
    //                             1,
    //                             A,
    //                             lda,
    //                             x,
    //                             m);
    // else if(std::is_same<T, double>::value)
    //     return rocblas_dtrsm(   handle,
    //                             rocblas_side_left,
    //                             uplo,
    //                             transA,
    //                             diag,
    //                             m,
    //                             1,
    //                             A,
    //                             lda,
    //                             x,
    //                             m);
    // else
    //     return rocblas_status_not_implemented;

    // if(std::is_same<T, float>::value)
    //     return status;
    // else if(std::is_same<T, double>::value)
    //     return status;
    // else
    //     return rocblas_status_not_implemented;





    //Kernel Call Here for Other Implementation

//    rocblas_status status = rocblas_trtri_trsm_template<T, BLOCK>(
//        handle, (T*)C_tmp.get(), uplo, diag, k, A, lda, (T*)invA.get());

// #define TRSV_DIM_X 96 //
// #define TRSV_DIM_Y 8 // TRSV_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
//         rocblas_int blocks = 1; /*(m - 1) / (TRSV_DIM_X * 4) + 1*/

//         dim3 trsv_grid(blocks, 1, 1);
//         dim3 trsv_threads(TRSV_DIM_X, TRSV_DIM_Y, 1);
//         hipStream_t rocblas_stream = handle->rocblas_stream;

//         if(handle->pointer_mode == rocblas_pointer_mode_device)
//         {
//             hipLaunchKernelGGL((trsv_kernel_device_pointer<T, TRSV_DIM_X, TRSV_DIM_Y>),
//                                dim3(trsv_grid),
//                                dim3(trsv_threads),
//                                (m+(m-BLOCK)*BLOCK+2*BLOCK*BLOCK)*sizeof(T),
//                                rocblas_stream,
//                                m,
//                                A,
//                                lda,
//                                x,
//                                incx);
//         }

// #undef TRSV_DIM_X
// #undef TRSV_DIM_Y

    return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocblas_strsv(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
                                        const float* A,
                                        rocblas_int lda,
                                        float* x,
                                        rocblas_int incx)
{
    return rocblas_trsv_template<float, STRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx);
}

extern "C" rocblas_status rocblas_dtrsv(rocblas_handle handle,
                                        rocblas_fill uplo,
                                        rocblas_operation transA,
                                        rocblas_diagonal diag,
                                        rocblas_int m,
                                        const double* A,
                                        rocblas_int lda,
                                        double* x,
                                        rocblas_int incx)
{
    return rocblas_trsv_template<double, DTRSV_BLOCK>(
        handle, uplo, transA, diag, m, A, lda, x, incx);
}
