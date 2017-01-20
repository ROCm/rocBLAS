/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/


#include <typeinfo>
#include "rocblas.h"
#include "rocblas.hpp"

/*!\file
 * \brief provide template functions interfaces to ROCBLAS C89 interfaces
*/



    /*
     * ===========================================================================
     *    level 1 BLAS
     * ===========================================================================
     */
    //scal
    template<>
    rocblas_status
    rocblas_scal<float>(rocblas_handle handle,
        rocblas_int n,
        const float *alpha,
        float *x, rocblas_int incx){

        return rocblas_sscal(handle, n, alpha, x, incx);
    }

    template<>
    rocblas_status
    rocblas_scal<double>(rocblas_handle handle,
        rocblas_int n,
        const double *alpha,
        double *x, rocblas_int incx){

        return rocblas_dscal(handle, n, alpha, x, incx);
    }

    template<>
    rocblas_status
    rocblas_scal<rocblas_float_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *alpha,
        rocblas_float_complex *x, rocblas_int incx){

        return rocblas_cscal(handle, n, alpha, x, incx);
    }

    template<>
    rocblas_status
    rocblas_scal<rocblas_double_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *alpha,
        rocblas_double_complex *x, rocblas_int incx){

        return rocblas_zscal(handle, n, alpha, x, incx);
    }

    //swap
    template<>
    rocblas_status
    rocblas_swap<float>(    rocblas_handle handle, rocblas_int n,
                            float *x, rocblas_int incx,
                            float *y, rocblas_int incy)
    {
        return rocblas_swap(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_swap<double>(   rocblas_handle handle, rocblas_int n,
                            double *x, rocblas_int incx,
                            double *y, rocblas_int incy)
    {
        return rocblas_dswap(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_swap<rocblas_float_complex>(    rocblas_handle handle, rocblas_int n,
                            rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *y, rocblas_int incy)
    {
        return rocblas_cswap(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_swap<rocblas_double_complex>(    rocblas_handle handle, rocblas_int n,
                            rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *y, rocblas_int incy)
    {
        return rocblas_zswap(handle, n, x, incx, y, incy);
    }

    //copy
    template<>
    rocblas_status
    rocblas_copy<float>(   rocblas_handle handle, rocblas_int n,
                            const float *x, rocblas_int incx,
                            float *y, rocblas_int incy)
    {
        return rocblas_scopy(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_copy<double>(   rocblas_handle handle, rocblas_int n,
                            const double *x, rocblas_int incx,
                            double *y, rocblas_int incy)
    {
        return rocblas_dcopy(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_copy<rocblas_float_complex>(    rocblas_handle handle, rocblas_int n,
                            const rocblas_float_complex *x, rocblas_int incx,
                            rocblas_float_complex *y, rocblas_int incy)
    {
        return rocblas_ccopy(handle, n, x, incx, y, incy);
    }

    template<>
    rocblas_status
    rocblas_copy<rocblas_double_complex>(    rocblas_handle handle, rocblas_int n,
                            const rocblas_double_complex *x, rocblas_int incx,
                            rocblas_double_complex *y, rocblas_int incy)
    {
        return rocblas_zcopy(handle, n, x, incx, y, incy);
    }

    //dot
    template<>
    rocblas_status
    rocblas_dot<float>(    rocblas_handle handle, rocblas_int n,
                            const float *x, rocblas_int incx,
                            const float *y, rocblas_int incy,
                            float *result)
    {
        return rocblas_sdot(handle, n, x, incx, y, incy, result);
    }

    template<>
    rocblas_status
    rocblas_dot<double>(    rocblas_handle handle, rocblas_int n,
                            const double *x, rocblas_int incx,
                            const double *y, rocblas_int incy,
                            double *result)
    {
        return rocblas_ddot(handle, n, x, incx, y, incy, result);
    }

    template<>
    rocblas_status
    rocblas_dot<rocblas_float_complex>(    rocblas_handle handle, rocblas_int n,
                            const rocblas_float_complex *x, rocblas_int incx,
                            const rocblas_float_complex *y, rocblas_int incy,
                            rocblas_float_complex *result)
    {
        return rocblas_cdotu(handle, n, x, incx, y, incy, result);
    }

    template<>
    rocblas_status
    rocblas_dot<rocblas_double_complex>(    rocblas_handle handle, rocblas_int n,
                            const rocblas_double_complex *x, rocblas_int incx,
                            const rocblas_double_complex *y, rocblas_int incy,
                            rocblas_double_complex *result)
    {
        return rocblas_zdotu(handle, n, x, incx, y, incy, result);
    }


    //asum
    template<>
    rocblas_status
    rocblas_asum<float, float>(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx,
        float *result){

        return rocblas_sasum(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_asum<double, double>(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx,
        double *result){

        return rocblas_dasum(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_asum<rocblas_float_complex, float>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *x, rocblas_int incx,
        float *result){

        return rocblas_scasum(handle, n, x, incx, result);
    }

    //nrm2
    template<>
    rocblas_status
    rocblas_nrm2<float, float>(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx,
        float *result){

        return rocblas_snrm2(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_nrm2<double, double>(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx,
        double *result){

        return rocblas_dnrm2(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_nrm2<rocblas_float_complex, float>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *x, rocblas_int incx,
        float *result){

        return rocblas_scnrm2(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_nrm2<rocblas_double_complex, double>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *x, rocblas_int incx,
        double *result){

        return rocblas_dznrm2(handle, n, x, incx, result);
    }


    //amin
    template<>
    rocblas_status
    rocblas_amin<float>(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_samin(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amin<double>(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_damin(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amin<rocblas_float_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_scamin(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amin<rocblas_double_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_dzamin(handle, n, x, incx, result);
    }

    //amax
    template<>
    rocblas_status
    rocblas_amax<float>(rocblas_handle handle,
        rocblas_int n,
        const float *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_samax(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amax<double>(rocblas_handle handle,
        rocblas_int n,
        const double *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_damax(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amax<rocblas_float_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_float_complex *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_scamax(handle, n, x, incx, result);
    }

    template<>
    rocblas_status
    rocblas_amax<rocblas_double_complex>(rocblas_handle handle,
        rocblas_int n,
        const rocblas_double_complex *x, rocblas_int incx,
        rocblas_int *result){

        return rocblas_dzamax(handle, n, x, incx, result);
    }

    /*
     * ===========================================================================
     *    level 2 BLAS
     * ===========================================================================
     */

    template<>
    rocblas_status
    rocblas_gemv<float>(    rocblas_handle handle,
                            rocblas_operation transA, rocblas_int m, rocblas_int n,
                            const float *alpha,
                            const float *A, rocblas_int lda,
                            const float *x, rocblas_int incx,
                            const float *beta, float *y, rocblas_int incy)
    {
        return rocblas_sgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template<>
    rocblas_status
    rocblas_gemv<double>(   rocblas_handle handle,
                            rocblas_operation transA, rocblas_int m, rocblas_int n,
                            const double *alpha,
                            const double *A, rocblas_int lda,
                            const double *x, rocblas_int incx,
                            const double *beta, double *y, rocblas_int incy)
    {
        return rocblas_dgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }


    /*
     * ===========================================================================
     *    level 3 BLAS
     * ===========================================================================
     */

    //



    //


