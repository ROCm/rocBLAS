
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
  * new unit tests should integrate within the existing [googletest framework](https://github.com/google/googletest/blob/master/googletest/docs/Primer.md)
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
-  [P.2](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-Cplusplus): Write in ISO Standard C++ (especially to support windows, linux and macos plaforms )
-  [P.5](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-compile-time): Prefer compile-time checking to run-time checking

### Implementation
-  [SF.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-file-suffix): Use a .cpp suffix for code files and .h for interface files if your project doesn't already follow another convention
  - We modify this rule:
    - .h: C header files
    - .hpp: C++ header files
-  [SF.5](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-consistency): A .cpp file must include the .h file(s) that defines its interface
-  [SF.7](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive): Don't put a using-directive in a header file
-  [SF.8](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-guards): Use #include guards for all .h files
-  [SF.21](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-unnamed): Don't use an unnamed (anonymous) namespace in a header
-  [SL.10](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rsl-arrays): Prefer using STL array or vector instead of a C array
-  [C.9](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-private): minimize exposure of members
-  [F.3](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-single): Keep functions short and simple
-  [F.21](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out-multi): To return multiple 'out' values, prefer returning a tuple
-  [R.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-raii): Manage resources automatically using RAII (this includes unique_ptr & shared_ptr)
-  [ES.11](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-auto):  use auto to avoid redundant repetition of type names
-  [ES.20](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-always): Always initialize an object
-  [ES.23](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-list): Prefer the {} initializer syntax
-  [ES.49](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-casts-named): If you must use a cast, use a named cast
-  [CP.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#S-concurrency): Assume that your code will run as part of a multi-threaded program
-  [I.2](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global): Avoid global variables
