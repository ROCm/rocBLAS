/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_reduction_strided_batched.hpp"

template <typename T>
void testing_iamax_strided_batched_bad_arg(const Arguments& arg)
{
    template_testing_reduction_strided_batched_bad_arg(arg, rocblas_iamax_strided_batched<T>);
}

template <typename T>
void testing_iamax_strided_batched(const Arguments& arg)
{
    template_testing_reduction_strided_batched(
        arg, rocblas_iamax_strided_batched<T>, rocblas_cblas::iamax<T>);
}

template <typename T>
void testing_iamin_strided_batched_bad_arg(const Arguments& arg)
{
    template_testing_reduction_strided_batched_bad_arg(arg, rocblas_iamin_strided_batched<T>);
}

template <typename T>
void testing_iamin_strided_batched(const Arguments& arg)
{
    template_testing_reduction_strided_batched(
        arg, rocblas_iamin_strided_batched<T>, rocblas_cblas::iamin<T>);
}
