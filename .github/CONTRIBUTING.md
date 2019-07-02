## Contribution License Agreement

1. The code I am contributing is mine, and I have the right to license it.

2. By submitting a pull request for this project I am granting you a license to distribute said code under the MIT License for the project.

## How to contribute

Our code contriubtion guidelines closely follows the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/).  This repository follows the [git flow](http://nvie.com/posts/a-successful-git-branching-model/) workflow, which dictates a /master branch where releases are cut, and a /develop branch which serves as an integration branch for new code.
  * A [git extention](https://github.com/nvie/gitflow) has been developed to ease the use of the 'git flow' methodology, but requires manual installation by the user.  Refer to the projects wiki

## Pull-request guidelines
* target the **develop** branch for integration
* ensure code builds successfully.
* do not break existing test cases
* new functionality will only be merged with new unit tests
  * new unit tests should integrate within the existing [googletest framework](https://github.com/google/googletest/blob/master/googletest/docs/primer.md)
  * tests must have good code coverage
  * code must also have benchmark tests, and performance must approach the compute bound limit or memory bound limit.

## StyleGuide
This project follows the [CPP Core guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md), with few modifications or additions noted below.  All pull-requests should in good faith attempt to follow the guidelines stated therein, but we recognize that the content is lengthy.  Below we list our primary concerns when reviewing pull-requests.

### Interface
-  All public APIs are C89 compatible; all other library code should use c++14
  - Our minimum supported compiler is clang 3.6
-  Avoid CamelCase
  - This rule applies specifically to publicly visible APIs, but is also encouraged (not mandated) for internal code

### Philosophy
-  [P.2](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-Cplusplus): Write in ISO Standard C++14 (especially to support windows, linux and macos plaforms )
-  [P.5](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-compile-time): Prefer compile-time checking to run-time checking

### Implementation
-  [SF.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-file-suffix): Use a `.cpp` suffix for code files and an `.h` suffix for interface files if your project doesn't already follow another convention
  - We modify this rule:
    - `.h`: C header files
    - `.hpp`: C++ header files
-  [SF.5](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-consistency): A `.cpp` file must include the `.h` file(s) that defines its interface
-  [SF.7](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive): Don't put a `using`-directive in a header file
-  [SF.8](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-guards): Use `#include` guards for all `.h` files
-  [SF.21](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-unnamed): Don't use an unnamed (anonymous) `namespace` in a header
-  [SL.10](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rsl-arrays): Prefer using `std::array` or `std::vector` instead of a C array
-  [C.9](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-private): Minimize the exposure of class members
-  [F.3](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-single): Keep functions short and simple
-  [F.21](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out-multi): To return multiple 'out' values, prefer returning a `std::tuple`
-  [R.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-raii): Manage resources automatically using RAII (this includes `std::unique_ptr` & `std::shared_ptr`)
-  [ES.11](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-auto): Use `auto` to avoid redundant repetition of type names
-  [ES.20](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-always): Always initialize an object
-  [ES.23](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-list): Prefer the `{}` initializer syntax
-  [CP.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#S-concurrency): Assume that your code will run as part of a multi-threaded program
-  [I.2](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global): Avoid global variables

## Format
C and C++ code is formatted using `clang-format`. Use the clang-format version for Clang 9, which is available in the `/opt/rocm` directory. Please do not use your system's built-in `clang-format`, as this is an older version that will result in different results.

To format a file, use:

```
/opt/rocm/hcc/bin/clang-format -style=file -i <path-to-source-file>
```

To format all files, run the following script in rocBLAS directory:

```
#!/bin/bash

find . \( -name build -o -name \*.git \) -prune -o \( -type f \( -name \*.h \
    -o -name \*.hpp -o -name \*.cpp -o -name \*.h.in -o -name \*.hpp.in -o \
    -name \*.cpp.in -o -name \*.cl \) -print0 \) | \
    xargs -0 /opt/rocm/hcc/bin/clang-format -i -style=file
```

Also, githooks can be installed to format the code per-commit:

```
./.githooks/install
```

## Here are some guidelines for writing rocBLAS code:

1. When writing function templates, place any non-type parameters before type parameters, i.e., leave the type parameters at the end. For example:

    ```c++
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
             return rocblas_trtri_small_batched<NB>(
                 handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count);
         else
             return rocblas_trtri_large_batched<NB>(
                 handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count, C_tmp);
    }
    ```
    The reason for this, is that the type template arguments can be automatically deduced from the actual function arguments, so that you don't have to pass the types in calls to the function, as shown in the example above when calling `rocblas_trtri_small_batched` and `rocblas_trtri_large_batched`. They have a `typename T` parameter too, but it can be automatically deduced, so it doesn't need to be explicitly passed.


2. When writing functions like the above which are heavily dependent on block sizes, especially if they are in header files included by other files, template parameters for block sizes are strongly preferred to `#define` macros or `constexpr` variables. For `.cpp` files which are not included in other files, a `constexpr` variable can be used. **Macros should never be used for constants.**


3. Code should not be copied-and pasted, but rather, templates, macros, [SFINAE](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error), [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern), [lambdas](https://en.cppreference.com/w/cpp/language/lambda), etc. should be used to factor out differences in similar code. A code should be made more generalized, rather than copied and modified. If a new function is similar to an existing function, then the new function and existing function should be refactored and based on a third templated function or class, rather than duplicating code.


4. `std::string` should only be used for strings whose length is unknown at compile time or which can grow. For simple static strings, or strings which are initialized once and then used read-only, `const char*` should be used to refer to the string or pass it as an argument. `std::string` involves dynamic memory allocation and copying of temporaries, which can be slow. `std::string_view` is supposed to help alleviate that, but it's not available until C++17, and we're using C++14 now. `const char*` should be used for read-only views of strings, in the interest of efficiency (an exception is made if the length is queried often, since `strlen()` is O(n)).


5. For most template functions which are used in other compilation units, it is preferred that they be put in header files, rather than `.cpp` files, because putting them in `.cpp` files requires explicit instantiation of them for all possible arguments, and there are less opportunities for inlining and interprocedural optimization. The C++ standard explicitly says that unused templates can be omitted from the output, so including unused templates in a header file does not increase the size of the program, since only the used ones are in the final output. For template functions which are only used in one `.cpp` file, they can be placed in the `.cpp` file. Templates, like inline functions, are granted an exception to the one definition rule (ODR) as long as the sequence of tokens in each compilation unit is identical.


6. Functions and namespace-scope variables which are not a part of the public interface of rocBLAS, should either be marked static, be placed in an unnamed namespace, or be placed in the `rocblas` namespace if they need to be accessible in other compilation units. For example:
    ```c++
    namespace
    {
        // Private internal implementation
    } // namespace

    extern "C"
    {
        // Public C interfaces
    } // extern "C"
    ```

7. If new arithmetic datatypes (like `rocblas_bfloat16`) are created, then unless they correspond *exactly* to a predefined system type, they should be wrapped into a `struct`, and not simply be a `typedef` to another type of the same size, so that their type is unique and can be differentiated from other types. Right now `rocblas_half` is `typedef`ed to `uint16_t`, which unfortunately prevents `rocblas_half` and `uint16_t` from being differentiable. If `rocblas_half` were simply a `struct` with a `uint16_t` member, then it would be a distinct type. It is legal to convert a pointer to a standard-layout `class`/`struct` to a pointer to its first element, and vice-versa.


8. [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization) classes should be used instead of explicit `new`/`delete`, `hipMalloc`/`hipFree`, `malloc`/`free`, etc. RAII classes are automatically exception-safe because their destructor gets called during unwinding. They only have to be declared once to construct them, and they are automatically destroyed when they go out of scope. This is better than having to match `new`/`delete` `malloc`/`free` calls in the code, especially when exceptions or early returns are possible. Even if the operation does not allocate and free memory, if it represents a change in state which must be undone when a function returns, then it belongs in an RAII class.


9. For code brevity and readability, when converting to numeric types, function-style casts are preferred to `static_cast<>()`. For example, `T(x)` is preferred to `static_cast<T>(x)`. When writing general containers or other classes which can accept *arbitrary* types as template parameters, not just *numeric* types, then the specific cast (`static_cast`, `const_cast`, `reinterpret_cast`) should be used, to avoid surprises. But when converting to *numeric* types, which have very well-understood behavior and are *side-effect free*, `type(x)` is more compact and clearer than `static_cast<type>(x)`. For pointers, C-style casts are okay, such as `(T*)A`.


10. With the new device memory allocation system, rocBLAS kernels should not call `hipMalloc() ` or `hipFree()` in their own code, but should use the new device memory manager. rocBLAS kernels in the public API must also detect and respond to device memory size queries. **There is another whole document on this.** TODO: Add link to device memory allocation document.


11. For BLAS2 functions and BLAS1 functions with two vectors, the `incx` and/or `incy` arguments can be negative, which means the vector is treated backwards from the end. A simple trick to handle this, is to adjust the pointer to the end of the vector if the increment is negative, as in:
    ```c++
    if(incx < 0)
        x -= ptrdiff_t(incx) * (n - 1);
    if(incy < 0)
        y -= ptrdiff_t(incy) * (n - 1);
    ```
    After that adjustment, the code does not need to treat negative increments any differently than positive ones.


12. To differentiate between scalars located on either the host or device memory, a special function has been created, called `load_scalar()`.
If its argument is a pointer, it is dereferenced on the device. If the argument is a scalar, it is simply copied. This allows single HIP kernels to be written for both device and host memory:
    ```c++
    template <typename T, typename U>
    __global__ void axpy_kernel(rocblas_int n, U alpha_device_host, const T* x, rocblas_int incx, T* y, rocblas_int incy)
    {
        auto alpha = load_scalar(alpha_device_host);
        ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

       // bound
       if(tid < n)
           y[tid * incy] += alpha * x[tid * incx];
    }
    ```
    Here, `alpha_device_host` can either be a pointer to device memory, or a numeric value passed directly to the kernel from the host. The `load_scalar()` function dereferences it if it's a pointer to device memory, and simply returns its argument if it's numerical. The kernel is called from the host in one of two ways depending on the pointer mode:
    ```c++
    if(handle->pointer_mode == rocblas_pointer_mode_device)
        hipLaunchKernelGGL(axpy_kernel, blocks, threads, 0, rocblas_stream, n, alpha, x, incx, y, incy);
    else if(*alpha) // alpha is on host
        hipLaunchKernelGGL(axpy_kernel, blocks, threads, 0, rocblas_stream, n, *alpha, x, incx, y, incy);
    ```
    When the pointer mode indicates `alpha` is on the host, the `alpha` pointer is dereferenced on the host and the numeric value it points to is passed to the kernel. When the pointer mode indicates `alpha` is on the device, the `alpha` pointer is passed to the kernel and dereferenced by the kernel on the device. This allows a single kernel to handle both cases, eliminating duplicate code.


13. For reduction operations, the file [reduction.h](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/src/blas1/reduction.h) has been created to systematize reductions and perform their device kernels in one place. This works for `amax`, `amin`, `asum`, `nrm2`, and (partially) `dot` and `gemv`. `rocblas_reduction_kernel` is a generalized kernel which takes 3 *functors* as template arguments: One to *fetch* values (such as fetching a complex value and taking the sum of the squares of its real and imaginary parts before reducing it), one to *reduce* values (such as to compute a sum or maximum), and one to *finalize* the reduction (such as taking the square root of a sum of squares). There is a `default_value()` function which returns the default value for a reduction. The default value is the value of the reduction when the size is 0, and reducing a value with the `default_value()` does not change the value of the reduction.


14. static duration variables which aren't constants should be made as function-local `static` variables, rather than as namespace or class static variables. This is to avoid the [static initialization order fiasco](https://isocpp.org/wiki/faq/ctors#static-init-order). For example:
    ```c++
    static auto& get_table()
    {
        // Placed inside function to avoid dependency on initialization order
        static table_t* table = test_cleanup::allocate<table_t>(table);
        return *table;
    }
    ```
    This is sometimes called the singleton pattern. A `static` variable is made local to a function rather than a namespace or class, and it gets initialized the first time the function is called. A reference to the `static` variable is returned from the function, and the function is used everywhere access to the variable is needed. In the case of multithreaded programs, the C++11 and later standards guarantee that there won't be any race conditions. It is also [faster](https://www.modernescpp.com/index.php/thread-safe-initialization-of-a-singleton) to initialize function-local `static` variables than it is to explicitly call `std::call_once`. For example:

    ```c++
    void my_func()
    {
        static int dummy = (func_to_call_once(), 0);
    }
    ```
    This is much faster than explicitly calling `std::call_once`, since the compiler has special ways of optimizing `static` initialization. The first time `my_func()` is called, it will call `func_to_call_once()` just once in a thread-safe way. After that, there is almost no overhead in later calls to `my_func()`.

15. Pointer mode should be switched to host mode during kernels which pass constants to other kernels, so that host-side constants of `-1.0`, `0.0` and `1.0` can be passed to kernels like `GEMM`. For example:
    ```c++
    // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    // Get alpha
    T alpha_h;
    if(saved_pointer_mode == rocblas_pointer_mode_host)
        alpha_h = *alpha;
    else
        RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost));
    ```
    `saved_pointer_mode` can be read to get the old pointer mode. If the old pointer mode was host pointer mode, then the host pointer is dereferenced to get the value of alpha. If the old pointer mode was device pointer mode, then the value of `alpha` is copied from the device to the host.

    After the above code switches to host pointer mode, constant values can be passed to `GEMM` or other kernels by always assuming host mode:
    ```c++
    static constexpr T negative_one = -1;
    static constexpr T zero = 0;
    static constexpr T one = 1;

    rocblas_gemm_template( handle, transA, transB, jb, n, jb, alpha, invA, BLOCK, B, ldb, &zero, X, m);
    ```
    When `saved_pointer_mode` is destroyed, the handle's pointer mode returns to the previous pointer mode.


16. When tests are added to `rocblas-test` and `rocblas-bench`, they should use [this guide](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/gtest/README.md).

    The test framework is templated, and uses [SFINAE](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error) to enable/disable certain types for certain tests for certain types. There are `std::enable_if<...>` expressions which enable certain combinations of types.


17. The `rocblas-test` and `rocblas-bench` [type dispatch file](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/clients/include/type_dispatch.hpp) is central to all tests. Basically, rather than duplicate:
    ```c++
    if(type == rocblas_datatype_f16_r)
        func1<rocblas_half>(args);
    else if(type == rocblas_datatype_f32_r)
        func<float>(args);
    else if(type == rocblas_datatype_f64_r)
        func<double>(args);
    ```
    etc. everywhere, it's done only in one place, and a `template` template argument is passed to specify which action is actually taken. It's fairly abstract, but it is powerful. There are examples of using the type dispatch in `clients/gtest/*_gtest.cpp` and `clients/benchmarks/client.cpp`.


18. Functions are preferred to macros. Functions or functors inside of `class` / `struct` templates can be used when partial template specializations are needed.

    When C preprocessor macros are needed (such as if they contain a `return` statement), if they are more than simple expressions, then [they should be wrapped in a `do { } while(0)`](https://stackoverflow.com/questions/154136/why-use-apparently-meaningless-do-while-and-if-else-statements-in-macros), without a terminating semicolon. This is to allow them to be used inside `if` statements.
