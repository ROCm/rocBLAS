# rocBLAS
rocBLAS is the AMD library for Basic Linear Algebra Subprograms (BLAS) on the [ROCm platform][1]. It is
implemented in the [HIP programming language][2] and optimized for AMD GPUs.

## Documentation
Information about the library API and other user topics can be found in the
[rocBLAS documentation][3].

## Prerequisites
The [AMD ROCm install guide][4] describes how to set up the ROCm repositories
and install the required platform dependencies.

## Installing pre-built packages
With the AMD ROCm package repositories installed, the `rocblas` package can be
retrieved from the system package manager. For example, on Ubuntu:

    sudo apt-get update
    sudo apt-get install rocblas

[1]: https://docs.amd.com
[2]: https://github.com/ROCm-Developer-Tools/HIP
[3]: https://rocblas.readthedocs.io/en/latest/
[4]: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
