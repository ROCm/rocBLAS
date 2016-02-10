/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ABLAS_COMMON_H_
#define _ABLAS_COMMON_H_

#include "ablas_types.h" 


/*!\file
 * \brief provide some common integer operations.
 */


    /* ============================================================================================ */
    /* integer functions */

    /*! \brief  For integers x >= 0, y > 0, returns ceil( x/y ).
     *          For x == 0, this is 0.
     */
    __host__ __device__
    static inline ablas_int ablas_ceildiv( ablas_int x, ablas_int y )
    {
        return (x + y - 1)/y;
    }

    /*! \brief  For integers x >= 0, y > 0, returns x rounded up to multiple of y.
     *          For x == 0, this is 0. y is not necessarily a power of 2.         
     */
    __host__ __device__
    static inline ablas_int ablas_roundup( ablas_int x, ablas_int y )
    {
        return ablas_ceildiv( x, y ) * y;
    }


#endif
