/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_iamax_iamin_ref.hpp"
#include "testing_reduction_batched.hpp"

template <typename T>
void testing_iamax_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_iamax_batched_fn
        = FORTRAN ? rocblas_iamax_batched<T, true> : rocblas_iamax_batched<T, false>;
    template_testing_reduction_batched_bad_arg(arg, rocblas_iamax_batched_fn);
}

template <typename T>
void testing_iamax_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_iamax_batched_fn
        = FORTRAN ? rocblas_iamax_batched<T, true> : rocblas_iamax_batched<T, false>;
    template_testing_reduction_batched(
        arg, rocblas_iamax_batched_fn, rocblas_iamax_iamin_ref::iamax<T>);
}

template <typename T>
void testing_iamin_batched_bad_arg(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_iamin_batched_fn
        = FORTRAN ? rocblas_iamin_batched<T, true> : rocblas_iamin_batched<T, false>;
    template_testing_reduction_batched_bad_arg(arg, rocblas_iamin_batched_fn);
}

template <typename T>
void testing_iamin_batched(const Arguments& arg)
{
    const bool FORTRAN = arg.fortran;
    auto       rocblas_iamin_batched_fn
        = FORTRAN ? rocblas_iamin_batched<T, true> : rocblas_iamin_batched<T, false>;
    template_testing_reduction_batched(
        arg, rocblas_iamin_batched_fn, rocblas_iamax_iamin_ref::iamin<T>);
}
