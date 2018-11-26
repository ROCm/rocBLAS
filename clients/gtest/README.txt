To add new data-driven tests to the ROCblas Google Test Framework:

1. Create a C++ file with the name:

       <function>_gtest.cpp

   ... where <function> is a non-type-specific shorthand for the function(s)
   being tested. Examples:

       gemm_gtest.cpp
       trsm_gtest.cpp
       blas1_gtest.cpp

   In the C++ file, perform these steps::

   A. #include the header files related to the tests, as well
      as test_dispatch.hpp. For example:

          #include "testing_syr.hpp"
          #include "type_dispatch.hpp"

   B. Wrap the body with an anonymous namespace, to minimize namespace
      collisions:

          namespace {

   C. Create a templated class which accepts any number of type parameters,
   followed by one anonymous trailing type parameter defaulted to void (to be
   used with enable_if).

   Choose the number of type parameters based on how likely in the future that
   the function will support a mixture of that many different types, e.g. Input
   type (Ti), Output type (To), Compute Type (Tc). If the function will never
   support more than 1-2 type parameters, then that many can be used.

   Make the primary definition of this class template derive from the
   rocblas_test_invalid class.

   For example:

       template <typename T, typename = void>
       struct syr_testing : rocblas_test_invalid
       {
       };

   D. Create one or more partial specializations of the class template,
      conditionally enabled by the type parameters matching legal combinations
      of types.

      If the first type argument is void, then these partial specializations
      need to not apply, so that the default based on rocblas_test_invalid can
      perform the correct behavior when void is passed to indicate failure.

      In the partial specialzation(s), create an explicit conversion to bool
      which returns true in the specialization. (By contrast,
      rocblas_test_invalid returns false.)

      In the partial specialization(s), create a functional operator() which
      takes a const Arguments& parameter and calls templated test functions
      (in include/testing_*.hpp) with the specialization's template arguments
      when the arg.function string matches the function name. If arg.function
      does not match any function related to this test, mark it a test failure.

      Example:

         template <typename T>
         struct syr_testing<T,
             typename std::enable_if<
                 std::is_same<T, float>::value ||
                 std::is_same<T, double>::value
             >::type
         >
         {
              explicit operator bool() { return true; }

              void operator()(const Arguments& arg)
              {
                  if(!strcmp(arg.function, "testing_syr"))
                      testing_syr<T>(arg);
                  else
                      FAIL()
                      << "Internal error: Test called with unknown function: "
                      << arg.function;
              }
         };

   E. If necessary, create a type dispatch function for this function (or class
      of functions it belongs to) in include/type_dispatch.hpp.

      The type dispatch function takes a "template" template parameter of
      template<typename...> class and a function parameter of type const
      Arguments&. It looks at the runtime type values in Arguments, and
      instantiates the template with one or more static type arguments,
      corresponding to the dynamic runtime type arguments.

      It treats the passed template as a functor, passing the Arguments
      argument to a particular instantiation of it.

      The combinations of types handled by this "runtime type to template
      type instantiation mapping" function can be general, because the type
      combinations which do not apply to a particular test case will have the
      template argument set to derive from rocblas_test_invalid, which will
      not create any unresolved instantiations. If unresolved instantiation
      compile or link errors occur, then the enable_if<> condition in step D
      needs to be refined to be false for type combinations which do not apply.

      The return type of this function needs to be variadic, picking up the
      return type of the functor.

      If the runtime type combinations do not apply, then this function should
      return TEST<void>()(arg), where TEST is the template parameter. However,
      this is less important than step D above in excluding invalid type
      combinations with enable_if, since this only excludes them at run-time,
      and they need to be excluded by step D at compile-time in order to avoid
      unresolved references or invalid instantiations.
      
      Example:

          template <template <typename...> class TEST>
          auto rocblas_simple_dispatch(const Arguments& arg)
              -> decltype(TEST<void>()(arg))
          {
              switch(arg.a_type)
              {
                case rocblas_datatype_f16_r: return TEST<rocblas_half>()(arg);
                case rocblas_datatype_f32_r: return TEST<float>()(arg);
                case rocblas_datatype_f64_r: return TEST<double>()(arg);
                case rocblas_datatype_f16_c: return TEST<rocblas_half_complex>()(arg);
                case rocblas_datatype_f32_c: return TEST<rocblas_float_complex>()(arg);
                case rocblas_datatype_f64_c: return TEST<rocblas_double_complex>()(arg);
                default: return TEST<void>()(arg);
               }
          }

   F. Create a (possibly-templated) test implemention class which derives from
      the RocBLAS_Test template class, passing itself to RocBLAS_Test (the CRTP
      pattern) as well as the template class defined above. Example:

          struct syr : RocBLAS_Test<syr, syr_testing>
          {

      In this class, implement three static functions:

      I. static bool type_filter(const Arguments& arg) returns true if the
      types described by *_type in the Arguments structure, match a valid
      type combination.

      This is usually implemented simply by calling the dispatch function in
      step E, passing it the helper type_filter_functor template class defined
      in RocBLAS_Test. This functor uses the same runtime type checks as are
      used to instantiate test functions with particular type arguments, but
      instead, this returns true or false depending on whether a real function
      would have been called. It is used to filter out tests whose runtime
      parameters do not match a valid test.

      Since RocBLAS_Test is a dependent base class if this test implementation
      class is templated, you may need to use a fully-qualified name (A::B) to
      resolve type_filter_functor, and in the last part of this name, the
      keyword "template" needs to precede type_filter_functor. The first half
      of the fully-qualified name can be this class itself, or the full
      instantation of RocBLAS_Test<...>. Example:

          static bool type_filter(const Arguments& arg)
          {
               return rocblas_blas1_dispatch<
                   blas1_test_template::template type_filter_functor>(arg);
          }

      II. static bool function_filter(const Arguments& arg) returns true if
      function name in Arguments matches one of the functions handled by this
      test. For example:

          // Filter for which functions apply to this suite
          static bool function_filter(const Arguments& arg)
          {
              return !strcmp(arg.function, "testing_ger") ||
                  !strcmp(arg.function, "testing_ger_bad_arg");
          }
                        
      III. static std::string name_suffix(const Arguments& arg) returns a
      string which will be used as the Google Test name's suffix. It will
      provide an alphanumeric representation of the test's arguments.

      The RocBLAS_TestName helper class template should be used to create the
      name. It accepts ostream output (like cout), and can be automatically
      converted to std::string after all of the text of the name has been
      streamed to it.

      The RocBLAS_TestName helper class template should be passed the name of
      this test implementation class (including any implicit template
      arguments), so that every instantiation of this test class creates a
      unique instantiation of RocBLAS_TestName. RocBLAS_TestName has some
      static data which needs to be local to each test.

      RocBLAS_TestName converts non-alphanumeric characters into suitable
      replacements, and disambiguates test names when the same arguments appear
      more than once.

      Since the conversion of the stream into a string is a destructive
      one-time operation, the RocBLAS_TestName value converted to std::string
      needs to be an rvalue. For example:
  
          static std::string name_suffix(const Arguments& arg)
          {
              // Okay: rvalue RocBLAS_TestName object streamed to and returned
              return RocBLAS_TestName<syr>() << rocblas_datatype2char(arg.a_type)
                  << '_' << (char) std::toupper(arg.uplo_option) << '_' << arg.N
		  << '_' << arg.alpha << '_' << arg.incx << '_' << arg.lda;
          }
 
          static std::string name_suffix(const Arguments& arg)
          {
              RocBLAS_TestName<gemm_test_template> name;

              name << rocblas_datatype2char(arg.a_type);
  
              if(GEMM_TYPE == GEMM_EX || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX)
                  name << rocblas_datatype2char(arg.b_type)
                      << rocblas_datatype2char(arg.c_type)
                      << rocblas_datatype2char(arg.d_type)
                      << rocblas_datatype2char(arg.compute_type);

              name << '_' << (char) std::toupper(arg.transA_option)
                  << (char) std::toupper(arg.transB_option) << '_' << arg.M 
                  << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_'
                  << arg.lda << '_' << arg.ldb << '_' << arg.beta << '_'
		  << arg.ldc;

              // name is an lvalue: Must use std::move to convert it to rvalue.
              // name cannot be used after it's converted to a string, which
              // is why it must be "moved" to a string.
              return std::move(name);
          }
            
    G. Choose a non-type-specific shorthand name for the test which will be
       displayed as part of the test name in the Google Tests output. Create a
       type alias for this name, unless the name is already the name of the
       class defined in step F, and it is not templated. For example, for a
       templated class defined in step F, create an alias for one of its
       instantiations:

           using gemm = gemm_test_template<gemm_testing, GEMM>;

    H. Pass the name created in step G to the TEST_P macro, along with a broad
       test category name that this test belongs to (so that Google Test
       filtering can be used to select all tests in a category).

       In the body following this TEST_P macro, call the dispatch function from
       step E, passing it the class from step C as a template template
       argument, and passing the result of GetParam() as an Arguments
       structure.

       For example:

           TEST_P(gemm, blas3) { rocblas_gemm_dispatch<gemm_testing>(GetParam()); }

    I. Call the macro which instantiates the Google Tests across all test
       categories (quick, pre_checkin, nightly, known_bug), passing it the same
       test name as in steps G and H. For example:

           INSTANTIATE_TEST_CATEGORIES(gemm);

    J. Don't forget to close the anonymous namespace:

        } // namespace

2. Create a <function>.yaml file with the same name as the C++ file, just with
   a .yaml extension.

   In the YAML file, define tests with combinations of parameters.

   The syntax and idioms of the YAML files is best described by looking at the
   existing *_gtest.yaml files as examples.

3. Add the YAML file to rocblas_gtest.yaml, to be included. For examnple:

       include: blas1_gtest.yaml

4. Add the YAML file to the list of dependencies for rocblas_gtest.data in
   CMakeLists.txt.  For example:

       add_custom_command( OUTPUT "${ROCBLAS_TEST_DATA}"
                           COMMAND ../common/rocblas_gentest.py -I ../include rocblas_gtest.yaml -o "${ROCBLAS_TEST_DATA}"
                           DEPENDS ../common/rocblas_gentest.py rocblas_gtest.yaml ../include/rocblas_common.yaml known_bugs.yaml blas1_gtest.yaml gemm_gtest.yaml gemv_gtest.yaml symv_gtest.yaml syr_gtest.yaml ger_gtest.yaml trsm_gtest.yaml trtri_gtest.yaml geam_gtest.yaml set_get_vector_gtest.yaml set_get_matrix_gtest.yaml
                           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

Many examples are available in gtest/*_gtest.{cpp.yaml}
