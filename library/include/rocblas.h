/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*!\file
 * \brief rocblas.h includes other *.h and exposes a common interface
 */

#ifndef _ROCBLAS_H_
#define _ROCBLAS_H_

/* Workaround clang bug:

   https://bugs.llvm.org/show_bug.cgi?id=35863

   This macro expands to static if clang is used; otherwise it expands empty.
   It is intended to be used in variable template specializations, where clang
   requires static in order for the specializations to have internal linkage,
   while technically, storage class specifiers besides thread_local are not
   allowed in template specializations, and static in the primary template
   definition should imply internal linkage for all specializations.

   If clang shows an error for improperly using a storage class specifier in
   a specialization, then ROCBLAS_CLANG_STATIC should be redefined as empty,
   and perhaps removed entirely, if the above bug has been fixed.
*/
#if __clang__
#define ROCBLAS_CLANG_STATIC static
#else
#define ROCBLAS_CLANG_STATIC
#endif

/* library headers */
#include "rocblas-auxiliary.h"
#include "rocblas-export.h"
#include "rocblas-functions.h"
#include "rocblas-types.h"
#include "rocblas-version.h"

#endif // _ROCBLAS_H_
