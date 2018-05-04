/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_POTF2_HPP
#define ROCLAPACK_POTF2_HPP


#include <hip/hip_runtime.h>

#include "rocblas.h"

#include "helpers.h"
#include "status.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

#include "../blas1/rocblas_dot.hpp"
#include "../blas1/rocblas_scal.hpp"
#include "../blas2/rocblas_gemv.hpp"
using namespace std;

template<typename T>
__global__ void sqrtDiagFirst(T* a, size_t loc, T* res){
    const T t = a[loc];
    if(t <= 0.0){res[3] = -loc;} // error for non-positive definiteness
    a[loc] = sqrt(t);
    res[1] = 1/a[loc];
}
    
template<typename T>
__global__ void sqrtDiagOnward(T* a, size_t loc, T* res){
    const T t = a[loc]-res[0];
    if(t <= 0.0){res[3] = -loc;} // error for non-positive definiteness
    a[loc] = sqrt(t);
}

template<typename T>
rocblas_status rocblas_potf2_template(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, T* a, rocblas_int lda){
        
    // store original pointer mode before setting it to device
    rocblas_pointer_mode pointer;
    rocblas_get_pointer_mode(handle, &pointer);    
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
    
    rocblas_int oneInt = 1;
    T inpsHost[3];
    inpsHost[0] = static_cast<T>(1.0f);
    inpsHost[1] = static_cast<T>(-1.0f);
        
    T resHost[4];
    resHost[3] = 0;
        
    // allocate a tiny bit of memory on device to avoid going onto CPU and needing to synchronize (eventually)
    T* results;
    T* inputs;
    hipMalloc(&results, 4 * sizeof(T));
    hipMalloc(&inputs, 3 * sizeof(T));
    hipMemcpy(inputs, &inpsHost[0], 2*sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(&results[3], &resHost[3], sizeof(T), hipMemcpyHostToDevice); // initialize error signal to 0

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
        
    if(n == 0) {
        // quick return
        return rocblas_status_success;
    } else if(n < 0){
        // less than zero dimensions in a matrix?!
        return rocblas_status_invalid_size;
    } else if(lda < max(1, n)){
        // mismatch of provided first matrix dimension
        return rocblas_status_invalid_size;
    }

    // in order to get the indices right, we check what the fill mode is
    if(uplo == rocblas_fill_upper){
            
    // Compute the Cholesky factorization A = U'*U.
           
    for(rocblas_int j = 0; j < n; ++j){
        // Compute U(J,J) and test for non-positive-definiteness.
        if(j > 0){
            rocblas_dot_template<T>(handle, j, &a[idx2D(0,j,lda)], oneInt, &a[idx2D(0,j,lda)], oneInt, &results[0]);
            //hipLaunchKernelGGL( sqrtDiagOnward<T>, dim3(1), dim3(1), 0, stream, a, idx2D(j,j,lda), results);
            hipMemcpy(&resHost[0], &results[0], sizeof(T), hipMemcpyDeviceToHost);
            } else {
                //hipLaunchKernelGGL( sqrtDiagFirst<T>, dim3(1), dim3(1), 0, stream, a, idx2D(j,j,lda), results);
                resHost[0] = 0.0;
            }
            hipMemcpy(&inpsHost[2], &a[idx2D(j,j,lda)], sizeof(T), hipMemcpyDeviceToHost);                
            resHost[1] = inpsHost[2] - resHost[0];
            if(resHost[1] <= 0.0){ // XXX I would like to have this in results and just check this at the very end...
                // something ain't right
                hipMemcpy(&a[idx2D(j,j,lda)], &resHost[1], sizeof(T), hipMemcpyHostToDevice);
                hipFree(results);
                hipFree(inputs);
                return rocblas_status_internal_error;
            }
                
            resHost[1] = sqrt(resHost[1]);
            hipMemcpy(&a[idx2D(j,j,lda)], &resHost[1], sizeof(T), hipMemcpyHostToDevice);
                
            // Compute elements J+1:N of row J
                
            if(j < n-1){
                rocblas_gemv_template<T>(handle, rocblas_operation_transpose, j, n-j-1, &(inputs[1]), &a[idx2D(0,j+1,lda)], lda, &a[idx2D(0,j,lda)], oneInt, &(inputs[0]), &a[idx2D(j,j+1,lda)], lda);
                resHost[2] = (inpsHost[0])/resHost[1];
                rocblas_scal_template<T>(handle, n-j-1, &resHost[2], &a[idx2D(j,j+1,lda)], lda);
            }
        }
            
    } else {
            
        // Compute the Cholesky factorization A = L'*L.
          
        for(rocblas_int j = 0; j < n; ++j){
            // Compute L(J,J) and test for non-positive-definiteness.
            if(j > 0){
                rocblas_dot_template<T>(handle, j, &a[idx2D(j,0,lda)], lda, &a[idx2D(j,0,lda)], lda, &results[0]);
                hipMemcpy(&resHost[0], &results[0], sizeof(T), hipMemcpyDeviceToHost);
            } else {
                resHost[0] = 0;
            }
            hipMemcpy(&inpsHost[2], &a[idx2D(j,j,lda)], sizeof(T), hipMemcpyDeviceToHost);
            resHost[1] = inpsHost[2] - resHost[0];
                
            if(resHost[1] <= 0.0){
                // something ain't right
                hipMemcpy(&a[idx2D(j,j,lda)], &resHost[1], sizeof(T), hipMemcpyHostToDevice);
                hipFree(results);
                hipFree(inputs);
                return rocblas_status_internal_error;
            }
                
            resHost[1] = sqrt(resHost[1]);
            hipMemcpy(&a[idx2D(j,j,lda)], &resHost[1], sizeof(T), hipMemcpyHostToDevice);
                
            // Compute elements J+1:N of row J
                
            if(j < n-1){
                rocblas_gemv_template<T>(handle, rocblas_operation_none, n-j-1, j, &(inputs[1]), &a[idx2D(j+1,0,lda)], lda, &a[idx2D(j,0,lda)], lda, &(inputs[0]), &a[idx2D(j+1,j,lda)], oneInt);
                resHost[2] = (inpsHost[0])/resHost[1];
                rocblas_scal_template<T>(handle, n-j-1, &resHost[2], &a[idx2D(j+1,j,lda)], oneInt);
            }
        }
    }

    // get the error code using memcpy and return internal error if there is one
    hipMemcpy(&resHost[3], &results[3], sizeof(T), hipMemcpyDeviceToHost);
    if(resHost[3] <= 0.0){
        const size_t elem = static_cast<size_t>(fabs(resHost[3]));
        cerr << "ERROR: Input matrix not strictly positive definite. Last occurance of this in element " << elem << endl;
        hipFree(results);
        hipFree(inputs);
        return rocblas_status_internal_error;
    }
        
    hipFree(results);
    hipFree(inputs);
    
    // restore pointer mode
    rocblas_set_pointer_mode(handle, pointer);
    
    return rocblas_status_success;
}

#endif /* ROCLAPACK_POTF2_HPP */

