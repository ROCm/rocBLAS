/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <hip_runtime.h>
#include "rocblas_types.h"
#include "rocblas_hip.h"

    /* ============================================================================================ */

    /*! \brief  indicates whether the pointer is on the host or device. currently HIP API can only recoginize the input ptr on deive or not
    can not recoginize it is on host or not */
    rocblas_pointer_type rocblas_get_pointer_type(void *ptr){
        hipPointerAttribute_t attribute;
        hipPointerGetAttributes(&attribute, ptr);
        //if( rocblas_success != (rocblas_status)hipPointerGetAttributes(&attribute, ptr) ){
        //    printf("failed to get the pointer type\n");
        //}
        if (ptr == attribute.devicePointer) {
            return DEVICE_POINTER;
        }
        else{
            return HOST_POINTER;
        }
    }


    /*! \brief   create rocblas handle called before any rocblas library routines*/
    extern "C"
    rocblas_status rocblas_create(rocblas_handle *handle){
        /* TODO */
        rocblas_int device;

        rocblas_status status;
        status = (rocblas_status)hipGetDevice(&device);// return the active device

        if (status != rocblas_success) {
            return status;
        }

        handle->device_id = device;
        return rocblas_success;
    }

    /*! \brief   release rocblas handle, will implicitly synchronize host and device */
    extern "C"
    rocblas_status rocblas_destroy(rocblas_handle handle){
        /* TODO */

        return rocblas_success;
    }


    /*! \brief   set rocblas stream used for all subsequent library function calls.
     *   If not set, all hip kernels will take the default NULL stream. stream_id must be created before this call */
    extern "C"
    rocblas_status
    rocblas_set_stream(rocblas_handle handle, hipStream_t stream_id){
        handle.stream = stream_id;
        return rocblas_success;
    }


    /*! \brief   get rocblas stream used for all subsequent library function calls.
     *   If not set, all hip kernels will take the default NULL stream. */
    extern "C"
    rocblas_status
    rocblas_get_stream(rocblas_handle handle, hipStream_t *stream_id){
        *stream_id = handle.stream;
        return rocblas_success;
    }



    /* ============================================================================================ */
    // synchronouse Functions
    /*! \brief   copy a vector hx of length n on host to a vector dx on device */
    extern "C"
    rocblas_status rocblas_set_vector(rocblas_int n,
                                      rocblas_int elem_size,
                                      const void *hx, rocblas_int incx,
                                      void *dx, rocblas_int incy){

        if ( n < 0 ){
            return  rocblas_invalid_dim;
        }
        else if(elem_size < 0){
            return  rocblas_invalid_size;
        }
        else if(hx == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( incx < 0 ){
            return  rocblas_invalid_incx;
        }
        else if(hx == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( incy < 0 ){
            return  rocblas_invalid_incy;
        }

        if(incx != 1 || incy != 1){
            printf("incx != incy ! = 1, rocblas_get_vector is not available now \n");
            return rocblas_not_implemented;
        }
        else{
            return (rocblas_status)hipMemcpy(dx, hx, elem_size * n, hipMemcpyHostToDevice);
        }
    }

    /*! \brief   copy a vector dx of length n on device to a vector hx on host */
    extern "C"
    rocblas_status rocblas_get_vector(rocblas_int n,
                                      rocblas_int elem_size,
                                      const void *dx, rocblas_int incx,
                                      void *hx, rocblas_int incy){

        if ( n < 0 ){
            return  rocblas_invalid_dim;
        }
        else if(elem_size < 0){
            return  rocblas_invalid_size;
        }
        else if(hx == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( incx < 0 ){
            return  rocblas_invalid_incx;
        }
        else if(hx == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( incy < 0 ){
            return  rocblas_invalid_incy;
        }

        if(incx != 1 || incy != 1){
            printf("incx != incy ! = 1, rocblas_get_vector is not available now \n");
            return rocblas_not_implemented;
        }
        else{
            return (rocblas_status)hipMemcpy(hx, dx, elem_size * n, hipMemcpyDeviceToHost);
        }
    }


    /*! \brief   copy row*column part of A on host to row*column part of B on device. Both A and B are in column-major */
    extern "C"
    rocblas_status rocblas_set_matrix(rocblas_int row, rocblas_int column,
                                      rocblas_int elem_size,
                                      const void *A, rocblas_int lda,
                                      void *B, rocblas_int ldb){

        if ( row < 0 ){
            return  rocblas_invalid_dim;
        }
        else if(column < 0){
            return  rocblas_invalid_dim;
        }
        else if(elem_size < 0){
            return  rocblas_invalid_size;
        }
        else if(A == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( lda < row ){
            return  rocblas_invalid_leadDimA;
        }
        else if(B == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( ldb < row ){
            return  rocblas_invalid_leadDimB;
        }


        if(row == lda && lda == ldb){
            return (rocblas_status)hipMemcpy(B, A, elem_size * row * column, hipMemcpyHostToDevice);
        }
        else{
            printf("row != lda ! = ldb, rocblas_set_matrix is not available now \n");
            return rocblas_not_implemented;
        }
    }



    /*! \brief   copy row*column part of A on device to row*column part of B on host. Both A and B are in column-major */
    extern "C"
    rocblas_status rocblas_get_matrix(rocblas_int row, rocblas_int column,
                                      rocblas_int elem_size,
                                      const void *A, rocblas_int lda,
                                      void *B, rocblas_int ldb){

        if ( row < 0 ){
            return  rocblas_invalid_dim;
        }
        else if(column < 0){
            return  rocblas_invalid_dim;
        }
        else if(elem_size < 0){
            return  rocblas_invalid_size;
        }
        else if(A == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( lda < row ){
            return  rocblas_invalid_leadDimA;
        }
        else if(B == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( ldb < row ){
            return  rocblas_invalid_leadDimB;
        }

        if(row == lda && lda == ldb){
            return (rocblas_status)hipMemcpy(B, A, elem_size * row * column, hipMemcpyDeviceToHost);
        }
        else{
            printf("row != lda ! = ldb, rocblas_get_matrix is not available now \n");
            return rocblas_not_implemented;
        }
    }


    /* ============================================================================================ */
    // asynchronouse Functions
    /*! \brief   copy a vector hx of length n on host to a vector dx on device. done asynchronously */
    extern "C"
    rocblas_status rocblas_set_vector_async(rocblas_int n,
                                      rocblas_int elem_size,
                                      const void *hx, rocblas_int incx,
                                      void *dx, rocblas_int incy, hipStream_t stream){


        if ( n < 0 ){
            return  rocblas_invalid_dim;
        }
        else if(elem_size < 0){
            return  rocblas_invalid_size;
        }
        else if(hx == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( incx < 0 ){
            return  rocblas_invalid_incx;
        }
        else if(hx == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( incy < 0 ){
            return  rocblas_invalid_incy;
        }

        if(incy == 1){
            if(incx == 1){
                return (rocblas_status)hipMemcpyAsync(dx, hx, elem_size * n, hipMemcpyHostToDevice, stream);
            }
            else{
                //malloc a buffer on CPU and copy hx to packed buffer, then hipMemcy buffer to dx.
                printf("incx ! = 1, rocblas_set_vector_async is not available now \n");
                return rocblas_not_implemented;
            }
        }
        else{
                printf("incy ! = 1, rocblas_set_vector_async is not available now \n");
                return rocblas_not_implemented;

        }
    }


    /*! \brief   copy a vector dx of length n on device to a vector hx on host. done asynchronously */
    extern "C"
    rocblas_status rocblas_get_vector_async(rocblas_int n,
                                      rocblas_int elem_size,
                                      const void *dx, rocblas_int incx,
                                      void *hx, rocblas_int incy, hipStream_t stream){


        if ( n < 0 ){
            return  rocblas_invalid_dim;
        }
        else if(elem_size < 0){
            return  rocblas_invalid_size;
        }
        else if(hx == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( incx < 0 ){
            return  rocblas_invalid_incx;
        }
        else if(hx == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( incy < 0 ){
            return  rocblas_invalid_incy;
        }

        if(incx != 1 || incy != 1){
            printf("incx != incy ! = 1, rocblas_get_vector_async is not available now \n");
            return rocblas_not_implemented;
        }
        else{
            return (rocblas_status)hipMemcpyAsync(hx, dx, elem_size * n, hipMemcpyDeviceToHost, stream);
        }
    }



    /*! \brief   copy row*column part of A on host to row*column part of B on device. Both A and B are in column-major. done asynchronously */
    extern "C"
    rocblas_status rocblas_set_matrix_async(rocblas_int row, rocblas_int column,
                                      rocblas_int elem_size,
                                      const void *A, rocblas_int lda,
                                      void *B, rocblas_int ldb, hipStream_t stream){


        if ( row < 0 ){
            return  rocblas_invalid_dim;
        }
        else if(column < 0){
            return  rocblas_invalid_dim;
        }
        else if(elem_size < 0){
            return  rocblas_invalid_size;
        }
        else if(A == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( lda < row ){
            return  rocblas_invalid_leadDimA;
        }
        else if(B == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( ldb < row ){
            return  rocblas_invalid_leadDimB;
        }

        if(row == lda && lda == ldb){
            return (rocblas_status)hipMemcpyAsync(B, A, elem_size * row * column, hipMemcpyHostToDevice, stream);
        }
        else{
            printf("row != lda ! = ldb, rocblas_set_matrix_async is not available now \n");
            return rocblas_not_implemented;
        }
    }


    /*! \brief   copy row*column part of A on device to row*column part of B on host. Both A and B are in column-major. done asynchronously */
    extern "C"
    rocblas_status rocblas_get_matrix_async(rocblas_int row, rocblas_int column,
                                      rocblas_int elem_size,
                                      const void *A, rocblas_int lda,
                                      void *B, rocblas_int ldb, hipStream_t stream){


        if ( row < 0 ){
            return  rocblas_invalid_dim;
        }
        else if(column < 0){
            return  rocblas_invalid_dim;
        }
        else if(elem_size < 0){
            return  rocblas_invalid_size;
        }
        else if(A == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( lda < row ){
            return  rocblas_invalid_leadDimA;
        }
        else if(B == NULL){
            return rocblas_error_invalid_value;
        }
        else if ( ldb < row ){
            return  rocblas_invalid_leadDimB;
        }

        if(row == lda && lda == ldb){
            return (rocblas_status)hipMemcpyAsync(B, A, elem_size * row * column, hipMemcpyDeviceToHost, stream);
        }
        else{
            printf("row != lda ! = ldb, rocblas_get_matrix_async is not available now \n");
            return rocblas_not_implemented;
        }
    }
