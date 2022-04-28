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

#pragma once

#include "rocblas_iamax_iamin_ref.hpp"
#include "testing_reduction_batched.hpp"

template <typename T>
void testing_iamax_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_iamax_batched_fn
        = arg.fortran ? rocblas_iamax_batched<T, true> : rocblas_iamax_batched<T, false>;
    template_testing_reduction_batched_bad_arg(arg, rocblas_iamax_batched_fn);
}

template <typename T>
void testing_iamax_batched(const Arguments& arg)
{
    auto rocblas_iamax_batched_fn
        = arg.fortran ? rocblas_iamax_batched<T, true> : rocblas_iamax_batched<T, false>;
    template_testing_reduction_batched(
        arg, rocblas_iamax_batched_fn, rocblas_iamax_iamin_ref::iamax<T>);
}

template <typename T>
void testing_iamin_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_iamin_batched_fn
        = arg.fortran ? rocblas_iamin_batched<T, true> : rocblas_iamin_batched<T, false>;
    template_testing_reduction_batched_bad_arg(arg, rocblas_iamin_batched_fn);
}

template <typename T>
void testing_iamin_batched(const Arguments& arg)
{
    auto rocblas_iamin_batched_fn
        = arg.fortran ? rocblas_iamin_batched<T, true> : rocblas_iamin_batched<T, false>;
    template_testing_reduction_batched(
        arg, rocblas_iamin_batched_fn, rocblas_iamax_iamin_ref::iamin<T>);
}
