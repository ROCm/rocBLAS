# rocBLAS
rocBLAS is AMD's library for [BLAS](http://www.netlib.org/blas/) on [ROCm](https://rocm.github.io/install.html).
It is implemented in the [HIP](https://github.com/ROCm-Developer-Tools/HIP)
programming language and optimized for AMD's GPUs.

|Acronym      | Expansion                                                   |
|-------------|-------------------------------------------------------------|
|**BLAS**     | **B**asic **L**inear **A**lgebra **S**ubprograms            |
|**ROCm**     | **R**adeon **O**pen **C**ompute platfor**m**                |
|**HIP**      | **H**eterogeneous-Compute **I**nterface for **P**ortability |

## Documentation
The latest rocBLAS documentation and API description can be found [here](https://rocblas.readthedocs.io/en/latest/).

## Prerequisites
* A ROCm enabled platform, more information at [https://rocm.github.io/install.html](https://rocm.github.io/install.html).
* Base software stack, which includes
  * [HIP](https://github.com/ROCm-Developer-Tools/HIP)

## Installing pre-built packages
rocBLAS can be installed on Ubuntu using
```
sudo apt-get update
sudo apt-get install rocblas
```

rocBLAS Debian packages can also be downloaded from the
[rocBLAS releases tag](https://github.com/ROCmSoftwarePlatform/rocBLAS/releases).
These may be newer than the package from apt-get.

## Building rocBLAS from source, exported functions, and additional information

For additional information, please consult the
[wiki](https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki)
