/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ROCBLAS_COMMON_H_
#define _ROCBLAS_COMMON_H_

#include "rocblas_types.h" 


/*!\file
 * \brief provide some common integer operations.
 */


    /* ============================================================================================ */
    /* integer functions */

    /*! \brief  For integers x >= 0, y > 0, returns ceil( x/y ).
     *          For x == 0, this is 0.
     */
    __host__ __device__
    static inline rocblas_int rocblas_ceildiv( rocblas_int x, rocblas_int y )
    {
        return (x + y - 1)/y;
    }

    /*! \brief  For integers x >= 0, y > 0, returns x rounded up to multiple of y.
     *          For x == 0, this is 0. y is not necessarily a power of 2.         
     */
    __host__ __device__
    static inline rocblas_int rocblas_roundup( rocblas_int x, rocblas_int y )
    {
        return rocblas_ceildiv( x, y ) * y;
    }


#endif
