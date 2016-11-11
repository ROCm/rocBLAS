
    /*
     * ===========================================================================
     *    This file provide common device function used in various BLAS routines
     * ===========================================================================
     */


#include "rocblas.h"
#include "fetch_template.h"

template<>
 __device__ float
fetch_real<float, float>(float A)
{
    return A;
}

template<>
 __device__ float
fetch_imag<float, float>(float A)
{
    return 0.0;
}


template<>
 __device__ double
fetch_real<double, double>(double A)
{
    return A;
}

template<>
 __device__ double
fetch_imag<double, double>(double A)
{
    return 0.0;
}


template<>
__device__ float
fetch_real<rocblas_float_complex, float>(rocblas_float_complex A)
{
    return A.x;
}

template<>
__device__ float
fetch_imag<rocblas_float_complex, float>(rocblas_float_complex A)
{
    return A.y;
}


template<>
__device__ double
fetch_real<rocblas_double_complex, double>(rocblas_double_complex A)
{
    return A.x;
}

template<>
__device__ double
fetch_imag<rocblas_double_complex, double>(rocblas_double_complex A)
{
    return A.y;
}
