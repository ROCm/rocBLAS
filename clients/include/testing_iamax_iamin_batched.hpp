/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_iamax_iamin_ref.hpp"
#include "testing_reduction_batched.hpp"

template <typename T>
void testing_iamax_batched_bad_arg(const Arguments& arg)
{
    template_testing_reduction_batched_bad_arg(arg, rocblas_iamax_batched<T>);
}

template <typename T>
void testing_iamax_batched(const Arguments& arg)
{
    template_testing_reduction_batched(
        arg, rocblas_iamax_batched<T>, rocblas_iamax_iamin_ref::iamax<T>);
}

template <typename T>
void testing_iamin_batched_bad_arg(const Arguments& arg)
{
    template_testing_reduction_batched_bad_arg(arg, rocblas_iamin_batched<T>);
}

template <typename T>
void testing_iamin_batched(const Arguments& arg)
{
    template_testing_reduction_batched(
        arg, rocblas_iamin_batched<T>, rocblas_iamax_iamin_ref::iamin<T>);
}
