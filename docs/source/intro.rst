
.. toctree::
   :maxdepth: 4
   :caption: Contents:

************
Introduction
************

rocBLAS is a `BLAS <http://www.netlib.org/blas/>`__ implementation on top of AMD's Radeon Open Compute `ROCm <https://rocm.github.io/install.html>`__ runtime and toolchains.
rocBLAS is implemented in the `HIP <https://github.com/ROCm-Developer-Tools/HIP>`__ programming language and optimized for AMD's latest
discrete GPUs.

======== =========
Acronym  Expansion
======== =========
**BLAS**    **B**\ asic **L**\ inear **A**\ lgebra **S**\ ubprograms
**ROCm**    **R**\ adeon **O**\ pen **C**\ ompute Platfor\ **m**
**HIP**     **H**\ eterogeneous-Compute **I**\ nterface for **P**\ ortability
======== =========


The aim of rocBLAS is to provide:

- functionality similar to Legacy BLAS, adapted to run on GPUs
- high performance robust implementation

rocBLAS is written in C++14 and HIP. It uses AMD's ROCm runtime to run on GPU devices.

The rocBLAS API is a thin C89 API using the `Hourglass Pattern <https://github.com/CppCon/CppCon2014/blob/master/Presentations/Hourglass%20Interfaces%20for%20C%2B%2B%20APIs/Hourglass%20Interfaces%20for%20C%2B%2B%20APIs%20-%20Stefanus%20Du%20Toit%20-%20CppCon%202014.pdf/>`_. It contains:

- [Level1]_, [Level2]_, and [Level3]_ `BLAS <http://www.netlib.org/blas/>`_ functions, with batched and strided_batched versions
- Extensions to Legacy BLAS, including functions for mixed precision
- Auxiliary functions
- Device Memory functions

rocBLAS array storage format is column major and one based. This is to maintain compatibility with the Legacy BLAS code which is written in Fortran.

rocBLAS calls AMD's library `Tensile <https://github.com/ROCmSoftwarePlatform/Tensile>`_ for Level 3 BLAS matrix multiplication.

rocBLAS is initialized by calling rocblas_create_handle and it is terminated by calling rocblas_destroy_handle. The rocblas_handle is persistent and it contains

- HIP stream
- temporary device work space
- mode for enabling or disabling logging (default is logging disabled)

rocBLAS functions run on the host and they call HIP to launch rocBLAS kernels that run on the device in a HIP stream. The kernels are asynchronous unless:

- the function returns a scalar result from device to host
- temporary device memory is allocated

In both cases above, the launch can be made asynchronous by:

- use rocblas_pointer_mode_device to keep the scalar result on the device. Note that it is only the following Level1 BLAS functions that return a scalar result: Xdot, Xdotu, Xnrm2, Xasum, iXamax, iXamin.

- use the provided device memory functions to allocate device memory that persists in the handle. Note that most rocBLAS functions do not allocate temporary device memory.

Before calling a rocBLAS function arrays must be copied to the device. Integer scalars like m, n, k are stored on the host. Floating point scalars like alpha and beta can be on host or device.

Error handling is by returning a rocblas_status. Functions conform to the Legacy BLAS argument checking.

Below is a simple example code for calling function rocblas_sscal.

.. code:: cpp

   #include <iostream>
   #include <vector>
   #include "hip/hip_runtime_api.h"
   #include "rocblas.h"

   using namespace std;

   int main()
   {
       rocblas_int n = 10240;
       float alpha = 10.0;

       vector<float> hx(n);
       vector<float> hz(n);
       float* dx;

       rocblas_handle handle;
       rocblas_create_handle(&handle);

       // allocate memory on device
       hipMalloc(&dx, n * sizeof(float));

       // Initial Data on CPU,
       srand(1);
       for( int i = 0; i < n; ++i )
       {
           hx[i] = rand() % 10 + 1;  //generate a integer number between [1, 10]
       }

       // copy array from host memory to device memory
       hipMemcpy(dx, hx.data(), sizeof(float) * n, hipMemcpyHostToDevice);

       // call rocBLAS function
       rocblas_status status = rocblas_sscal(handle, n, &alpha, dx, 1);

       // check status for errors
       if(status == rocblas_status_success)
       {
           cout << "status == rocblas_status_success" << endl;
       }
       else
       {
           cout << "rocblas failure: status = " << status << endl;
       }

       // copy output from device memory to host memory
       hipMemcpy(hx.data(), dx, sizeof(float) * n, hipMemcpyDeviceToHost);

       hipFree(dx);
       rocblas_destroy_handle(handle);
       return 0;
   }

.. rubic:: References

.. [Level1] C. L. Lawson, R. J. Hanson, D. Kincaid, and F. T. Krogh, Basic Linear Algebra Subprograms for FORTRAN usage, ACM Trans. Math. Soft., 5 (1979), pp. 308--323.

.. [Level2] J. J. Dongarra, J. Du Croz, S. Hammarling, and R. J. Hanson, An extended set of FORTRAN Basic Linear Algebra Subprograms, ACM Trans. Math. Soft., 14 (1988), pp. 1--17

.. [Level3] J. J. Dongarra, J. Du Croz, S. Hammarling, and R. J. Hanson, Algorithm 656: An extended set of FORTRAN Basic Linear Algebra Subprograms, ACM Trans. Math. Soft., 14 (1988), pp. 18--32
