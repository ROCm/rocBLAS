# rocBLAS
# adding this line for testing
rocBLAS is the AMD library for [BLAS][1] on the [ROCm platform][2]. It is
implemented in the [HIP programming language][3] and optimized for AMD GPUs.

|Acronym      | Expansion                                                   |
|-------------|-------------------------------------------------------------|
|**BLAS**     | **B**asic **L**inear **A**lgebra **S**ubprograms            |
|**ROCm**     | **R**adeon **O**pen E**C**osyste**m**                       |
|**HIP**      | **H**eterogeneous-Compute **I**nterface for **P**ortability |

## Documentation
Information about the library API and other user topics can be found in the
[rocBLAS documentation][4].

## Prerequisites
The [AMD ROCm install guide][5] describes how to set up the ROCm repositories
and install the required platform dependencies.

## Installing pre-built packages
With the AMD ROCm package repositories installed, the `rocblas` package can be
retrieved from the system package manager. For example, on Ubuntu:

    sudo apt-get update
    sudo apt-get install rocblas

[1]: https://www.netlib.org/blas/
[2]: https://rocmdocs.amd.com/en/latest/
[3]: https://github.com/ROCm-Developer-Tools/HIP
[4]: https://rocblas.readthedocs.io/en/latest/
[5]: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
