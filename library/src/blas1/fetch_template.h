
/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

#pragma once
#ifndef _FETCH_TEMPLATE_
#define _FETCH_TEMPLATE_

template <typename T1, typename T2>
__device__ T2 fetch_real(T1 A);

template <typename T1, typename T2>
__device__ T2 fetch_imag(T1 A);

#endif
