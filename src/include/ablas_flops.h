/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

#pragma once
#ifndef _ABLAS_FLOPS_H_
#define _ABLAS_FLOPS_H_

#include <ablas_types.h>

/*!\file
 * \brief ablas_gflops.h provides Floating point counts of Basic Linear Algebra Subprograms of Level 1, 2 and 3
*/



#ifdef __cplusplus
extern "C" {
#endif

    /*
     * ===========================================================================
     *    level 2 BLAS
     * ===========================================================================
     */

    /* \brief floating point counts of GEMV */
    double  sgemv_gflops(ablas_int m, ablas_int n){
        return (double)(2 * m * n)/1e9;
    }

    double  dgemv_gflops(ablas_int m, ablas_int n){
        return (double)(2 * m * n)/1e9;
    }

    double  cgemv_gflops(ablas_int m, ablas_int n){
        return (double)(4 * 2 * m * n)/1e9;
    }

    double  zgemv_gflops(ablas_int m, ablas_int n){
        return (double)(4 * 2 * m * n)/1e9;
    }

    /* \brief floating point counts of SY(HE)MV */
    double  ssymv_gflops(ablas_int n){
        return (double)(2 * n * n)/1e9;
    }

    double  dsymv_gflops(ablas_int n){
        return (double)(2 * n * n)/1e9;
    }

    double  chemv_gflops(ablas_int n){
        return (double)(4 * 2 * n * n)/1e9;
    }

    double  zhemv_gflops(ablas_int n){
        return (double)(4 * 2 * n * n)/1e9;
    }

    /* \brief floating point counts of GEMM */
    double  sgemm_gflops(ablas_int m, ablas_int n, ablas_int k){
        return (double)(2 * m * n * k)/1e9;
    }

    double  dgemm_gflops(ablas_int m, ablas_int n, ablas_int k){
        return (double)(2 * m * n * k)/1e9;
    }

    double  cgemm_gflops(ablas_int m, ablas_int n, ablas_int k){
        return (double)(8 * m * n * k)/1e9;
    }

    double  zgemm_gflops(ablas_int m, ablas_int n, ablas_int k){
        return (double)(8 * m * n * k)/1e9;
    }

#ifdef __cplusplus
}
#endif

#endif  /* _ABLAS_FLOPS_H_ */
