/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
/* ************************************************************************
 * Copyright (c) 1992-2022 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved.
 * Copyright (c) 2000-2022 The University of California Berkeley. All
                        rights reserved.
 * Copyright (c) 2006-2022 The University of Colorado Denver.  All rights
                        reserved.

 * $COPYRIGHT$

 * Additional copyrights may follow

 * $HEADER$

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

 * The copyright holders provide no reassurances that the source code
 * provided does not infringe any patent, copyright, or any other
 * intellectual property rights of third parties.  The copyright holders
 * disclaim any liability to any recipient for claims brought against
 * recipient by any third party for infringement of that parties
 * intellectual property rights.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ************************************************************************ */
#pragma once

#include "rocblas_vector.hpp"

/* LAPACK library functionality */
template <typename T>
void lapack_xcombssq(T* ssq, T* colssq)
{
    if(ssq[0] >= colssq[0])
    {
        if(ssq[0] != 0)
        {
            ssq[1] = ssq[1] + std::sqrt(colssq[0] / ssq[0]) * colssq[1];
        }
        else
        {
            ssq[1] = ssq[1] + colssq[1];
        }
    }
    else
    {
        ssq[1] = colssq[1] + std::sqrt(ssq[0] / colssq[0]) * ssq[1];
        ssq[0] = colssq[0];
    }
    return;
}

/*! \brief  lapack_xlassq computes the scale and sumsq*/
template <typename T>
void lapack_xlassq(rocblas_int n, T* X, rocblas_int incx, double& scale, double& sumsq)
{
    if(n > 0)
    {
        double abs_X = 0.0;
        for(int i = 0; i < n; i++)
        {
            abs_X = rocblas_abs(rocblas_is_complex<T> ? std::real(X[i * incx]) : X[i * incx]);
            if(abs_X > 0 || rocblas_isnan(abs_X))
            {
                if(scale < abs_X)
                {
                    sumsq = 1 + sumsq * std::sqrt(scale / abs_X);
                    scale = abs_X;
                }
                else
                {
                    sumsq = sumsq + std::sqrt(abs_X / scale);
                }
            }
            if(rocblas_is_complex<T>)
            {
                abs_X = rocblas_abs(std::imag(X[i * incx]));
                if(abs_X > 0 || rocblas_isnan(abs_X))
                {
                    if(scale < abs_X || rocblas_isnan(abs_X))
                    {
                        sumsq = 1 + sumsq * std::sqrt(scale / abs_X);
                        scale = abs_X;
                    }
                    else
                    {
                        sumsq = sumsq + std::sqrt(abs_X / scale);
                    }
                }
            }
        }
    }
}
/*! \brief lapack_xlange-returns the value of the one norm,  or the Frobenius norm, or the  infinity norm of the matrix A.*/
template <typename T>
double
    lapack_xlange(char norm_type, rocblas_int m, rocblas_int n, T* A, rocblas_int lda, double* work)
{
    double value = 0.0;
    double sum   = 0.0;
    if(std::min(m, n) == 0)
        return value;

    else if(norm_type == 'O' || norm_type == 'o' || norm_type == '1')
    {
        //Find the one norm of Matrix A.
        for(int j = 0; j < n; j++)
        {
            sum = 0.0;
            for(int i = 0; i < m; i++)
                sum = sum + rocblas_abs(A[i + j * lda]);

            if(value < sum || rocblas_isnan(sum))
                value = sum;
        }
    }
    else if(norm_type == 'I' || norm_type == 'i')
    {
        //Find the infinity norm of Matrix A.
        for(int j = 0; j < n; j++)
            for(int i = 0; i < m; i++)
            {
                work[i] = work[i] + rocblas_abs(A[i + j * lda]);
            }
        for(int i = 0; i < m; i++)
            if(value < work[i] || rocblas_isnan(work[i]))
                value = work[i];
    }
    else if(norm_type == 'F' || norm_type == 'f')
    {
        //Find the Frobenius norm of Matrix A.
        //SSQ(1) is scale
        //SSQ(2) is sum-of-squares
        //For better accuracy, sum each column separately.
        host_vector<double> ssq(2);
        host_vector<double> colssq(2);
        ssq[0] = 0.0;
        ssq[1] = 1.0;
        for(int j = 0; j < n; j++)
        {
            colssq[0] = 0.0;
            colssq[1] = 1.0;
            lapack_xlassq(m, A + j * lda, 1, colssq[0], colssq[1]);
            lapack_xcombssq(ssq.data(), colssq.data());
        }
        value = ssq[0] * std::sqrt(ssq[1]);
    }
    return value;
}
