## Questions, investigations, notes
-  With HC grid launch (GL), is it possible for a user to write \__KERNEL code, and then pass that function pointer to the library which could be called by a library kernel?  We need functionality like this for FFT, where users may provide pre- and post- callback functions the library invokes before/after the FFT kernels, respectively.

-  How do we handle the current requirement of clFFT to use run-time kernel compilation?  There does not seem to be a way in Kalmar currently to be able to JIT compile C/C++ code at runtime


Current notes on bring up library features and their compiler/runtime features:

- hcBLAS
  - library to provide HSA GEMM calls only
  - utilize CLOC to compile kernels at build time to link into library
  - utilize assembler to compile kernels at build time to link into library

- hcFFT
  - library to provide support for HSA 1d FFT’s complex-complex, size < 4096 only
  - utilize runtime HSA compiler features to build generated kernels at runtime and receive kernel handle (hsa_code_object_t?) which can be enqueued into a hsa_queue_t
  - API to accept user defined \__KERNEL pre-callback function written by the user and called inside FFT kernels through a functor or function pointer mechanism

- hcSPARSE
  - library to provide HSA Sparse matrix-vector multiply  using the Kalmar lambda kernel launch mechanism
  - use the Kalmar parallel STL to provide csr matrix transposes (counting iterators, sort_by_key, gather, zip iterators )

- hcRNG
  - library to provide MRG31k3p RNG generator in a single stream
  - library provides both host and device APIs; host to allocate blocks of random numbers in main memory, device to generate on demand random numbers per work-item
  - Currently, device API is implemented as OpenCL header files (\*.clh) \#included in user kernels.  Is there an alternate/better way to do this with HSA?
