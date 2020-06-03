============
Contributing
============

License Agreement
=================

1. The code I am contributing is mine, and I have the right to license
   it.

2. By submitting a pull request for this project I am granting you a
   license to distribute said code under the MIT License for the
   project.

Pull-request guidelines
=======================


Our code contriubtion guidelines closely follows the model of `GitHub
pull-requests <https://help.github.com/articles/using-pull-requests/>`__.
The rocBLAS repository follows the `git
flow <http://nvie.com/posts/a-successful-git-branching-model/>`__
workflow, which dictates a /master branch where releases are cut, and a
/develop branch which serves as an integration branch for new code. Pull requests should:

-  target the **develop** branch for integration
-  ensure code builds successfully.
-  do not break existing test cases
-  new functionality will only be merged with new unit tests
-  new unit tests should integrate within the existing `googletest
   framework <https://github.com/google/googletest/blob/master/googletest/docs/primer.md>`__
-  tests must have good code coverage
-  code must also have benchmark tests, and performance must approach
   the compute bound limit or memory bound limit.

StyleGuide
==========

This project follows the `CPP Core
guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`__,
with few modifications or additions noted below. All pull-requests
should in good faith attempt to follow the guidelines stated therein,
but we recognize that the content is lengthy. Below we list our primary
concerns when reviewing pull-requests.

Interface
---------

-  All public APIs are C89 compatible; all other library code should use
   c++14
-  Our minimum supported compiler is clang 3.6
-  Avoid CamelCase
-  This rule applies specifically to publicly visible APIs, but is also
   encouraged (not mandated) for internal code

Philosophy
----------

-  `P.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-Cplusplus>`__:
   Write in ISO Standard C++14 (especially to support windows, linux and
   macos plaforms )
-  `P.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-compile-time>`__:
   Prefer compile-time checking to run-time checking

Implementation
--------------

-  `SF.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-file-suffix>`__:
   Use a ``.cpp`` suffix for code files and an ``.h`` suffix for
   interface files if your project doesn't already follow another
   convention
-  We modify this rule:

   -  ``.h``: C header files
   -  ``.hpp``: C++ header files

-  `SF.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-consistency>`__:
   A ``.cpp`` file must include the ``.h`` file(s) that defines its
   interface
-  `SF.7 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive>`__:
   Don't put a ``using``-directive in a header file
-  `SF.8 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-guards>`__:
   Use ``#include`` guards for all ``.h`` files
-  `SF.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-unnamed>`__:
   Don't use an unnamed (anonymous) ``namespace`` in a header
-  `SL.10 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rsl-arrays>`__:
   Prefer using ``std::array`` or ``std::vector`` instead of a C array
-  `C.9 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-private>`__:
   Minimize the exposure of class members
-  `F.3 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-single>`__:
   Keep functions short and simple
-  `F.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out-multi>`__:
   To return multiple 'out' values, prefer returning a ``std::tuple``
-  `R.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-raii>`__:
   Manage resources automatically using RAII (this includes
   ``std::unique_ptr`` & ``std::shared_ptr``)
-  `ES.11 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-auto>`__:
   Use ``auto`` to avoid redundant repetition of type names
-  `ES.20 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-always>`__:
   Always initialize an object
-  `ES.23 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-list>`__:
   Prefer the ``{}`` initializer syntax
-  `CP.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#S-concurrency>`__:
   Assume that your code will run as part of a multi-threaded program
-  `I.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global>`__:
   Avoid global variables

Format
------

C and C++ code is formatted using ``clang-format``. To run clang-format
use the version in the ``/opt/rocm`` directory. Please do not use your 
system's built-in ``clang-format``, as this may be an older version that 
will result in different results.

To format a file, use:

::

    /opt/rocm/hcc/bin/clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in rocBLAS directory:

::

    #!/bin/bash
    git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/hcc/bin/clang-format  -style=file -i

Also, githooks can be installed to format the code per-commit:

::

    ./.githooks/install

Coding Guidelines
=================

1.  With the `rocBLAS device memory allocation
    system <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/docs/Device_Memory_Allocation.pdf>`__,
    rocBLAS kernels should not call ``hipMalloc()`` or ``hipFree()`` in
    their own code, but should use the device memory manager.

    ``hipMalloc()`` and ``hipFree()`` are synchronizing operations which
    should be avoided as much as possible.

    The device memory allocation system provides:

    -  A ``device_malloc`` method for temporarily using device memory
       which has either been allocated before, or which is allocated on
       demand.
    -  A method to reuse device memory across rocBLAS calls, without
       allocating them and deallocating them at every call.
    -  A method for users to query how much device memory is needed for
       a particular kernel call, in order for it to perform optimally.
    -  A method for users to control how much device memory is
       allocated, or whether to leave it up to rocBLAS to allocate it on
       demand.

    **Extra pointers or size arguments for temporary storage should not
    be added to the end of public APIs, only private internal ones.**
    Instead, implementations of the public APIs should request and
    obtain device memory using the rocBLAS device memory manager.
    rocBLAS kernels in the public API must also detect and respond to
    *device memory size queries*.

    A kernel must allocate all of its device memory upfront, for use
    during the entirety of the kernel call. It must not allocate and
    deallocate device memory at different levels of kernel calls. This
    means that if a lower-level kernel needs device memory, it must be
    allocated by higher-level routines and passed down to the
    lower-level routines. When device memory can be shared between two
    or more operations, the maximum size needed by all them should be
    reported or allocated.

    Details are in the `Device Memory
    Allocation <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/docs/Device_Memory_Allocation.pdf>`__
    design document. Examples of how to use the device memory allocator
    are in
    `TRSV <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas2/rocblas_trsv.cpp>`__
    and
    `TRSM <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas3/rocblas_trsm.cpp>`__.

2.  Logging, argument error checking and device memory allocation should
    only occur at the top-level kernel routines. Therefore, if one
    rocBLAS routine calls another, the lower-level called routine(s)
    should not perform logging, argument checking, or device memory
    allocation. This can be accomplished in one of two ways:

    A. (Preferred.) Abstract out the computational part of the kernel
    into a separate template function (usually named
    ``rocblas_<kernel>_template``, and call it from a higher-level
    template routine (usually named ``rocblas_<kernel>_impl``) which
    does error-checking, device memory allocation, and logging, and
    which gets called by the C wrappers:

    .. code:: cpp

        template <...>
        rocblas_status rocblas_<kernel>_template(..., T* device_memory)
        {
            // Performs fast computation
            // No argument error checking
            // No logging
            // No device memory allocation -- any temporary device memory must be passed in through pointers
            // Can be called by other computational kernels
            // Called by rocblas_<kernel>_impl
            // Private internal API
        }

        template <...>
        rocblas_status rocblas_<kernel>_impl()
        {
            // Argument error checking
            // Logging
            // Responding to device memory size queries
            // Device memory allocation (through handle->device_malloc())
            // Temporarily switching to host pointer mode if scalar constants are used
            // Calls rocblas_<kernel>_template()
            // Private internal API
        }

        extern "C" rocblas_status rocblas_[hsdcz]<kernel>()
        {
            // C wrapper
            // Calls rocblas_<kernel>_impl()
            // Public API
        }

    B. Use a ``bool`` template argument to specify if the kernel
    template should perform full functionality or not. Pass device
    memory pointer(s) which will be used if full functionality is turned
    off:

    .. code:: cpp

        template <bool full_function, ...>
        rocblas_status rocblas_<kernel>_template(..., T* device_memory = nullptr)
        {
            if(full_function)
            {
                // Argument error checking
                // Logging
                // Responding to device memory size queries
                // Device memory allocation (memory pointer assumed already allocated otherwise)*
                // Temporarily switching to host pointer mode if scalar constants are used*
                return rocblas_<kernel>_template<false, ...>(...);
            }
            // Perform fast computation
            // Private internal API
        }

    \*Device memory allocation, and temporarily switching pointer mode,
    might be difficult to enclose in an ``if`` statement with the RAII
    design, so the code might have to use recursion to call the
    non-fully-functional version of itself after setting these things
    up. That's why method A above is preferred, but for some huge
    functions like GEMM, method B might be more practical to implement,
    since it disrupts existing code less.

3.  The pointer mode should be temporarily switched to host mode during
    kernels which pass constants to other kernels, so that host-side
    constants of ``-1.0``, ``0.0`` and ``1.0`` can be passed to kernels
    like ``GEMM``, without causing synchronizing host<->device memory
    copies. For example:

    .. code:: cpp

        // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        // Get alpha
        T alpha_h;
        if(saved_pointer_mode == rocblas_pointer_mode_host)
            alpha_h = *alpha;
        else
            RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost));

    ``saved_pointer_mode`` can be read to get the old pointer mode. If
    the old pointer mode was host pointer mode, then the host pointer is
    dereferenced to get the value of alpha. If the old pointer mode was
    device pointer mode, then the value of ``alpha`` is copied from the
    device to the host.

    After the above code switches to host pointer mode, constant values
    can be passed to ``GEMM`` or other kernels by always assuming host
    mode:

    .. code:: cpp

        static constexpr T negative_one = -1;
        static constexpr T zero = 0;
        static constexpr T one = 1;

        rocblas_gemm_template( handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero, X, m);

    When ``saved_pointer_mode`` is destroyed, the handle's pointer mode
    returns to the previous pointer mode.

4.  When tests are added to ``rocblas-test`` and ``rocblas-bench``,
    refer to `this
    guide <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/gtest/README.md>`__.

    The test framework is templated, and uses
    `SFINAE <https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error>`__
    and ``std::enable_if<...>`` to enable and disable certain types for
    certain tests.

    YAML files are used to describe tests as combinations of arguments.
    ```rocblas_gentest.py`` <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/common/rocblas_gentest.py>`__
    is used to parse the YAML files and generate tests in the form of a
    binary file of
    ```Arguments`` <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/include/rocblas_arguments.hpp>`__
    records.

    The ``rocblas-test`` and ``rocblas-bench`` `type dispatch
    file <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/include/type_dispatch.hpp>`__
    is central to all tests. Basically, rather than duplicate:

    .. code:: cpp

        if(type == rocblas_datatype_f16_r)
            func1<rocblas_half>(args);
        else if(type == rocblas_datatype_f32_r)
            func<float>(args);
        else if(type == rocblas_datatype_f64_r)
            func<double>(args);

    etc. everywhere, it's done only in one place, and a ``template``
    template argument is passed to specify which action is actually
    taken. It's fairly abstract, but it is powerful. There are examples
    of using the type dispatch in
    ```clients/gtest/*_gtest.cpp`` <https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/develop/clients/gtest>`__
    and
    ```clients/benchmarks/client.cpp`` <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/benchmarks/client.cpp>`__.

5.  Code should not be copied-and pasted, but rather, templates, macros,
    `SFINAE <https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error>`__,
    `CRTP <https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern>`__,
    `lambdas <https://en.cppreference.com/w/cpp/language/lambda>`__,
    etc. should be used to factor out differences in similar code.

    A code should be made more generalized, rather than copied and
    modified, unless it is a completely different kernel function, and
    the old code is just being used as a start.

    If a new function is similar to an existing function, then the
    existing function should be generalized, or the new function and
    existing function should be refactored and based on a third
    templated function or class, rather than duplicating code.

6.  To differentiate between scalars located on either the host or
    device memory, a special function has been created, called
    ``load_scalar()``. If its argument is a pointer, it is dereferenced
    on the device. If the argument is a scalar, it is simply copied.
    This allows single HIP kernels to be written for both device and
    host memory:

    .. code:: cpp

        template <typename T, typename U>
        __global__ void axpy_kernel(rocblas_int n, U alpha_device_host, const T* x, rocblas_int incx, T* y, rocblas_int incy)
        {
            auto alpha = load_scalar(alpha_device_host);
            ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

           // bound
           if(tid < n)
               y[tid * incy] += alpha * x[tid * incx];
        }

    Here, ``alpha_device_host`` can either be a pointer to device
    memory, or a numeric value passed directly to the kernel from the
    host. The ``load_scalar()`` function dereferences it if it's a
    pointer to device memory, and simply returns its argument if it's
    numerical. The kernel is called from the host in one of two ways
    depending on the pointer mode:

    .. code:: cpp

        if(handle->pointer_mode == rocblas_pointer_mode_device)
            hipLaunchKernelGGL(axpy_kernel, blocks, threads, 0, rocblas_stream, n, alpha, x, incx, y, incy);
        else if(*alpha) // alpha is on host
            hipLaunchKernelGGL(axpy_kernel, blocks, threads, 0, rocblas_stream, n, *alpha, x, incx, y, incy);

    When the pointer mode indicates ``alpha`` is on the host, the
    ``alpha`` pointer is dereferenced on the host and the numeric value
    it points to is passed to the kernel. When the pointer mode
    indicates ``alpha`` is on the device, the ``alpha`` pointer is
    passed to the kernel and dereferenced by the kernel on the device.
    This allows a single kernel to handle both cases, eliminating
    duplicate code.

7.  If new arithmetic datatypes (like ``rocblas_bfloat16``) are created,
    then unless they correspond *exactly* to a predefined system type,
    they should be wrapped into a ``struct``, and not simply be a
    ``typedef`` to another type of the same size, so that their type is
    unique and can be differentiated from other types.

    Right now ``rocblas_half`` is ``typedef``\ ed to ``uint16_t``, which
    unfortunately prevents ``rocblas_half`` and ``uint16_t`` from being
    differentiable. If ``rocblas_half`` were simply a ``struct`` with a
    ``uint16_t`` member, then it would be a distinct type.

    It is legal to convert a pointer to a `standard-layout
    ``class``/``struct`` <https://en.cppreference.com/w/cpp/language/data_members#Standard_layout>`__
    to a pointer to its first element, and vice-versa, so the C API is
    unaffected by whether the type is enclosed in a ``struct`` or not.

8.  `RAII <https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization>`__
    classes should be used instead of explicit ``new``/``delete``,
    ``hipMalloc``/``hipFree``, ``malloc``/``free``, etc. RAII classes
    are automatically exception-safe because their destructor gets
    called during unwinding. They only have to be declared once to
    construct them, and they are automatically destroyed when they go
    out of scope. This is better than having to match ``new``/``delete``
    ``malloc``/``free`` calls in the code, especially when exceptions or
    early returns are possible.

    Even if an operation does not allocate and free memory, if it
    represents a change in state which must be undone when a function
    returns, then it belongs in an RAII class. For example,
    ``handle->push_pointer_mode()`` creates an RAII object which saves
    the pointer mode on construction, and restores it on destruction.

9.  When writing function templates, place any non-type parameters
    before type parameters, i.e., leave the type parameters at the end.
    For example:

    .. code:: cpp

        template <rocblas_int NB, typename T> // T is at end
        static rocblas_status rocblas_trtri_batched_template(rocblas_handle handle,
                                                             rocblas_fill uplo,
                                                             rocblas_diagonal diag,
                                                             rocblas_int n,
                                                             const T* A,
                                                             rocblas_int lda,
                                                             rocblas_int bsa,
                                                             T* invA,
                                                             rocblas_int ldinvA,
                                                             rocblas_int bsinvA,
                                                             rocblas_int batch_count,
                                                             T* C_tmp)
        {
            if(!n || !batch_count)
                return rocblas_status_success;

             if(n <= NB)
                 return rocblas_trtri_small_batched<NB>(  // T is automatically deduced
                     handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
             else
                 return rocblas_trtri_large_batched<NB>(  // T is automatically deduced
                     handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count, C_tmp);
        }

    The reason for this, is that the type template arguments can be
    automatically deduced from the actual function arguments, so that
    you don't have to pass the types in calls to the function, as shown
    in the example above when calling ``rocblas_trtri_small_batched``
    and ``rocblas_trtri_large_batched``. They have a ``typename T``
    parameter too, but it can be automatically deduced, so it doesn't
    need to be explicitly passed.

10. When writing functions like the above which are heavily dependent on
    block sizes, especially if they are in header files included by
    other files, template parameters for block sizes are strongly
    preferred to ``#define`` macros or ``constexpr`` variables. For
    ``.cpp`` files which are not included in other files, a
    ``static constexpr`` variable can be used. **Macros should never be
    used for constants.**

    Note: For constants inside of functions, ``static constexpr`` is
    preferred to just ``constexpr``, so that the variables do not need
    to be initialized at runtime.

    Note: C++14 variable templates can sometimes be used to provide
    constants. For example:

    .. code:: cpp

        template <typename T>
        static constexpr T negative_one = -1;

        template <typename T>
        static constexpr T zero = 0;

        template <typename T>
        static constexpr T one = 1;

11. static duration variables which aren't constants should usually be
    made function-local ``static`` variables, rather than namespace or
    class static variables. This is to avoid the `static initialization
    order
    fiasco <https://isocpp.org/wiki/faq/ctors#static-init-order>`__. For
    example:

    .. code:: cpp

        static auto& get_table()
        {
            // Placed inside function to avoid dependency on initialization order
            static std::unordered_map<std::string, size_t>* table = test_cleanup::allocate(&table);
            return *table;
        }

    This is sometimes called the *singleton* pattern. A ``static``
    variable is made local to a function rather than a namespace or
    class, and it gets initialized the first time the function is
    called. A reference to the ``static`` variable is returned from the
    function, and the function is used everywhere access to the variable
    is needed. In the case of multithreaded programs, the C++11 and
    later standards guarantee that there won't be any race conditions.
    It is also
    `faster <https://www.modernescpp.com/index.php/thread-safe-initialization-of-a-singleton>`__
    to initialize function-local ``static`` variables than it is to
    explicitly call ``std::call_once``. For example:

    .. code:: cpp

        void my_func()
        {
            static int dummy = (func_to_call_once(), 0);
        }

    This is much simpler and faster than explicitly calling
    ``std::call_once``, since the compiler has special ways of
    optimizing ``static`` initialization. The first time ``my_func()``
    is called, it will call ``func_to_call_once()`` once in a
    thread-safe way. After that, there is almost no overhead in later
    calls to ``my_func()``.

12. Functions are preferred to macros. Functions or functors inside of
    ``class`` / ``struct`` templates can be used when partial template
    specializations are needed.

    When C preprocessor macros are needed (such as if they contain a
    ``return`` statement to return from the calling function), if the
    macro's definition contains more than one simple expression, then
    `it should be wrapped in a
    ``do { } while(0)`` <https://stackoverflow.com/questions/154136/why-use-apparently-meaningless-do-while-and-if-else-statements-in-macros>`__,
    without a terminating semicolon. This is to allow them to be used
    inside ``if`` statements. For example:

    .. code:: cpp

        #define RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(h) \
            do                                               \
            {                                                \
                if((h)->is_device_memory_size_query())       \
                    return rocblas_status_size_unchanged;    \
            } while(0)

    The ``do { } while(0)`` allows the macro expansion to be a single
    statement which can be terminated with a semicolon, and which can be
    used anywhere a regular function call can be used.

13. For most template functions which are used in other compilation
    units, it is preferred that they be put in header files, rather than
    ``.cpp`` files, because putting them in ``.cpp`` files requires
    explicit instantiation of them for all possible arguments, and there
    are less opportunities for inlining and interprocedural
    optimization.

    The C++ standard explicitly says that unused templates can be
    omitted from the output, so including unused templates in a header
    file does not increase the size of the program, since only the used
    ones are in the final output.

    For template functions which are only used in one ``.cpp`` file,
    they can be placed in the ``.cpp`` file.

    Templates, like inline functions, are granted an exception to the
    one definition rule (ODR) as long as the sequence of tokens in each
    compilation unit is identical.

14. Functions and namespace-scope variables which are not a part of the
    public interface of rocBLAS, should either be marked static, be
    placed in an unnamed namespace, or be placed in
    ``namespace rocblas``. For example:

    .. code:: cpp

        namespace
        {
            // Private internal implementation
        } // namespace

        extern "C"
        {
            // Public C interfaces
        } // extern "C"

    However, unnamed namespaces should not be used in header files. If
    it is absolutely necessary to mark a function or variable as private
    to a compilation unit but defined in a header file, it should be
    declared ``static``, ``constexpr`` and/or ``inline`` (``constexpr``
    implies ``static`` for non-template variables and ``inline`` for
    functions).

    Even though rocBLAS goes into a shared library which exports a
    limited number of symbols, this is still a good idea, to decrease
    the chances of name collisions *inside* of rocBLAS.

15. ``std::string`` should only be used for strings which can grow, or
    which must be dynamically allocated as read-write strings. For
    simple static strings, strings returned from functions like
    ``getenv()``, or strings which are initialized once and then used
    read-only, ``const char*`` should be used to refer to the string or
    pass it as an argument.

    ``std::string`` involves dynamic memory allocation and copying of
    temporaries, which can be slow. ``std::string_view`` is supposed to
    help alleviate that, but it's not available until C++17, and we're
    using C++14 now. ``const char*`` should be used for read-only views
    of strings, in the interest of efficiency.

16. For code brevity and readability, when converting to *numeric*
    types, function-style casts are preferred to ``static_cast<>()`` or
    C-style casts. For example, ``T(x)`` is preferred to
    ``static_cast<T>(x)`` or ``(T)x``.

    When writing general containers or templates which can accept
    *arbitrary* types as parameters, not just *numeric* types, then the
    specific cast (``static_cast``, ``const_cast``,
    ``reinterpret_cast``) should be used, to avoid surprises.

    But when converting to *numeric* types, which have very
    well-understood behavior and are *side-effect free*, ``type(x)`` is
    more compact and clearer than ``static_cast<type>(x)``. For
    pointers, C-style casts are okay, such as ``(T*)A``.

17. For BLAS2 functions and BLAS1 functions with two vectors, the
    ``incx`` and/or ``incy`` arguments can be negative, which means the
    vector is treated backwards from the end. A simple trick to handle
    this, is to adjust the pointer to the end of the vector if the
    increment is negative, as in:

    .. code:: cpp

        if(incx < 0)
            x -= ptrdiff_t(incx) * (n - 1);
        if(incy < 0)
            y -= ptrdiff_t(incy) * (n - 1);

    After that adjustment, the code does not need to treat negative
    increments any differently than positive ones.

    Note: Some blocked matrix-vector algorithms which call other BLAS
    kernels may not work if this simple transformation is used; see
    `TRSV <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas2/rocblas_trsv.cpp>`__
    for an example, and how it's handled there.

18. For reduction operations, the file
    `reduction.h <https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas1/reduction.h>`__
    has been created to systematize reductions and perform their device
    kernels in one place. This works for ``amax``, ``amin``, ``asum``,
    ``nrm2``, and (partially) ``dot`` and ``gemv``.
    ``rocblas_reduction_kernel`` is a generalized kernel which takes 3
    *functors* as template arguments:

    -  One to *fetch* values (such as fetching a complex value and
       taking the sum of the squares of its real and imaginary parts
       before reducing it)
    -  One to *reduce* values (such as to compute a sum or maximum)
    -  One to *finalize* the reduction (such as taking the square root
       of a sum of squares)

    There is a ``default_value()`` function which returns the default
    value for a reduction. The default value is the value of the
    reduction when the size is 0, and reducing a value with the
    ``default_value()`` does not change the value of the reduction.

19. When `type punning <https://en.wikipedia.org/wiki/Type_punning>`__
    is needed, ``union`` should be used instead of pointer-casting,
    which violates *strict aliasing*. For example:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            union
            {
                uint32_t int32;
                float    fp32;
            } u = {uint32_t(data) << 16};
            return u.fp32; // Legal in C, nonstandard extension in C++
        }

    This violates the strict aliasing rule of
    `C <https://en.cppreference.com/w/c/language/object#Strict_aliasing>`__
    and
    `C++ <https://en.cppreference.com/w/cpp/language/reinterpret_cast#Type_aliasing>`__:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            uint32_t int32 = uint32_t(data) << 16;
            return *(float *) &int32; // Violates strict aliasing rule in both C and C++
        }

    The only 100% standard C++ way to do it, is to use ``memcpy()``, but
    this should not be required as long as GCC or Clang are used:

    .. code:: cpp

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        explicit __host__ __device__ operator float() const
        {
            uint32_t int32 = uint32_t(data) << 16;
            float fp32;
            static_assert(sizeof(int32) == sizeof(fp32), "Different sizes");
            memcpy(&fp32, &int32, sizeof(fp32));
            return fp32;
        }

20. ``<type_traits>`` classes which return Boolean values can be
    converted to ``bool`` in Boolean contexts. Hence many traits can be
    tested by simply creating an instance of them with ``{}``
    initialization syntax and using it in a Boolean context:

    .. code:: cpp

        template<typename T, typename = typename std::enable_if<std::is_same<T, float>{} ||
                                                                std::is_same<T, double>{}>::type>
        void function(T x)
        {
        }

    Here, instances of the ``std::is_same<...>`` traits class are
    created with the ``{}`` syntax. The resulting temporary objects can
    be explicitly converted to ``bool``, which is what occurs when an
    object appears in a conditional expression (``if``, ``while``,
    ``for``, ``&&``, ``||``, ``!``, ``? :``, etc.). This is a shorter
    syntax than using ``std::is_same<...>::value``.

21. ``rocblas_cout`` and ``rocblas_cerr`` should be used instead of ``std::cout``, ``std::cerr``, ``stdout`` or ``stderr``, and ``rocblas_ostream`` should be used instead of ``std::ostream``, ``std::ofstream`` or ``std::ostringstream``.

    In ``rocblas-bench`` and ``rocblas-test``, ``std::cout``, ``std::cerr``, ``printf``, ``fprintf``, ``stdout``, ``stderr``, ``puts()``, ``fputs()``, and other symbols are "poisoned", to remind you to use ``rocblas_cout``, ``rocblas_cerr``, and ``rocblas_ostream`` instead.

    ``rocblas_cout`` and ``rocblas_cerr`` are instances of ``rocblas_ostream`` which output to standard output and standard error, but in a way that prevents interlacing of different threads' output.

    ``rocblas_ostream`` provides standardized thread-safe formatted output for rocBLAS datatypes. It can be constructed in 3 ways:
    - By default, in which case it behaves like a ``std::ostringstream``
    - With a file descriptor number, in which case the file descriptor is ``dup()``ed and the same file it points to is outputted to
    - With a string, in which case a new file is opened for writing, with file creation, truncation and appending enabled (``O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC``)

    ``std::endl`` or ``std::flush`` should be used at the end of an output sequence when an atomic flush of the output is needed (atomic meaning that multiple threads can be writing to the same file, but that their flushes will be atomic). Until then, the output will accumulate in the ``rocblas_ostream`` and will not be flushed until either ``rocblas_ostream::flush()`` is called, ``std::endl`` or ``std::flush`` is outputted, or the ``rocblas_ostream`` is destroyed.

    The ``rocblas_ostream::yaml_on`` and ``rocblas_ostream::yaml_off`` IO modifiers enable or disable YAML formatting, for when outputting abitrary types as YAML source code. For example, to output a ``key: value`` pair as YAML source code, you would use:

    .. code:: cpp
        os << key << ": " << rocblas_ostream::yaml_on << value << rocblas_ostream::yaml_off;

    The ``key`` is outputted normally as a bare string, but the ``value`` uses YAML metacharacters and lexical syntax to output the value, so that when it's read in as YAML, it has the type and value of ``value``.


22. C++ templates, including variadic templates, are preferred to macros or runtime interpreting of values, although it is understood that sometimes macros are necessary.

    For example, when creating a class which models zero or more rocBLAS kernel arguments, it is preferable to use:
    .. code:: cpp

        template<rocblas_argument... Args>
        class ArgumentModel
        {
    public:
            void func()
            {
                for (auto arg: { Args... })
                {
                    // do something with argument arg
                }
            }
        };

        ArgumentModel<e_A, e_B>{}.func();

   instead of:

   .. code:: cpp

       class ArgumentModel
       {
            std::vector<rocblas_argument> args;
    public:
            ArgumentModel(const std::vector<rocblas_argument>& args)
            : args(args)
            {
            }

            void func()
            {
                for (auto arg: args)
                {
                    // do something with argument arg
                }
            }
        };

        ArgumentModel model({e_A, e_B});
        model.func();

  The former denotes the rocBLAS arguments as a list which is passed as a variadic template argument, and whose properties are known and can be optimized at compile-time, and which can be passed on as arguments to other templates, while the latter requires creating a dynamically-allocated runtime object which must be interpreted at runtime, such as by using ``switch`` statements on the arguments. The ``switch`` statement will need to list out and handle every possible argument, while the template solution simply passes the argument as another template argument, and hence can be resolved at compile-time.


23. Automatically-generated files should always go into ``build/`` directories, and should not go into source directories (even if marked ``.gitignore``). The CMake philosophy is such that you can create any ``build/`` directory, run ``cmake`` from there, and then have a self-contained build environment which will not touch any files outside of it.


24. The ``library/include`` subdirectory of rocBLAS, to be distinguished from the ``library/src/include`` subdirectory, shall consist only of C-compatible header files for public rocBLAS APIs. It should not include internal APIs, even if they are used in other projects, e.g., rocSOLVER, and the headers must be compilable with a C compiler, and must use ``.h`` extensions.


25. Macro parameters should only be evaluated once when practical, and should be parenthesized if there is a chance of ambiguous precedence. They should be stored in a local temporary variable if needed more than once.

Macros which expand to code with local variables, should use double-underscore suffixes in the local variable names, to prevent their conflict with variables passed in macro parameters. However, if they are in a completely separate block scope than the macro parameter is expanded in, or if they are only passed to another macro/function, then they do not need to use trailing underscores.

    ..code:: cpp

        #define CHECK_DEVICE_ALLOCATION(ERROR)                   \
            do                                                   \
            {                                                    \
                /* Use error__ in case ERROR contains "error" */ \
                hipError_t error__ = (ERROR);                    \
                if(error__ != hipSuccess)                        \
                {                                                \
                    if(error__ == hipErrorOutOfMemory)           \
                        SUCCEED() << LIMITED_MEMORY_STRING;      \
                    else                                         \
                        FAIL() << hipGetErrorString(error__);    \
                    return;                                      \
                }                                                \
            } while(0)

The ``ERROR`` macro parameter is evaluated only once, and is stored in the temporary variable ``error__``, for use multiple times later.

The ``ERROR`` macro parameter is parenthesized when initializing ``error__``, to avoid ambiguous precedence, such as if ``ERROR`` contains a comma expression.

The ``error__`` variable name is used, to prevent it from conflicting with variables passed in the ``ERROR`` macro parameter, such as ``error``.


26. Do not use variable-length arrays (VLA), which allocate on the stack, for arrays of unknown size.

    ..code:: cpp

        Ti* hostA[batch_count];
        Ti* hostB[batch_count];
        To* hostC[batch_count];
        To* hostD[batch_count];

        func(hostA, hostB, hostC, hostD);

 Instead, allocate on the heap, using smart pointers to avoid memory leaks:

    ..code:: cpp

        auto hostA = std::make_unique<Ti*[]>(batch_count);
        auto hostB = std::make_unique<Ti*[]>(batch_count);
        auto hostC = std::make_unique<To*[]>(batch_count);
        auto hostD = std::make_unique<To*[]>(batch_count);

        func(&hostA[0], &hostB[0], &hostC[0], &hostD[0]);


27. Do not define unnamed (anonymous) namespaces in header files `DCL59-CPP <https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL59-CPP.+Do+not+define+an+unnamed+namespace+in+a+header+file>`__

If the reason for using an unnamed namespace in a header file is to prevent multiple definitions, keep in mind that the following are allowed to be defined in multiple compilation units, such as if they all come from the same header file, as long as they are defined with identical token sequences in each compilation unit:

  -  ``class``es
  -  ``typedef``s or type aliases
  -  ``enum``s
  -  ``template`` functions
  -  ``inline`` functions
  -  ``constexpr`` functions (implies ``inline``)
  -  ``inline`` or ``constexpr`` variables or variable ``template``s (only for C++17 or later, although some C++14 compilers treat ``constexpr`` variables as ``inline``)

If functions defined in header files are declared ``template``, then multiple instantiations with the same ``template`` arguments are automatically merged, something which cannot happen if the ``template`` functions are declared ``static``, or appear in unnamed namespaces, in which case the instantiations are local to each compilation unit, and are not combined.

If a function defined in a header file at ``namespace`` scope (outside of a ``class``) contains ``static`` _local variables_ which are expected to be singletons holding state throughout the entire library, then the function cannot be marked ``static`` or be part of an unnamed ``namespace``, because then each compilation unit will have its own separate copy of that function and its local ``static`` variables. (``static`` member functions of classes always have external linkage, and it is okay to define ``static`` ``class`` member functions in-place inside of header files, because all in-place ``static`` member function definitions, including their ``static`` local variables, will be automatically merged.)

Guidelines:

-  Do not use unnamed ``namespace``s inside of header files.

-  Use either ``template`` or ``inline`` (or both) for functions defined outside of classes in header files.

-  Do not declare namespace-scope (not ``class``-scope) functions ``static`` inside of header files unless there is a very good reason, that the function does not have any non-``const`` ``static`` local variables, and that it is acceptable that each compilation unit will have its own independent definition of the function and its ``static`` local variables. (``static`` ``class`` member functions defined in header files are okay.)

-  Use ``static`` for ``constexpr`` ``template`` variables until C++17, after which ``constexpr`` variables become ``inline`` variables, and thus can be defined in multiple compilation units. It is okay if the ``constexpr`` variables remain ``static`` in C++17; it just means there might be a little bit of redundancy between compilation units.
