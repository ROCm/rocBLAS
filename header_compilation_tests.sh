#!/bin/bash

set -e
exec >&2

if [[ ! -e build/release/include/rocblas-export.h ]]; then
    echo "Please run this script after at least one build of rocBLAS."
    exit 1
fi

script=$(realpath "$0")

echocmd()
{
    cat <<EOF
-------------------------------------------------------------------------------
$@
EOF
    "$@"
}

hip_warning()
{
    cat <<EOF

<hip/hip_runtime.h> and (sometimes due to bugs) <hip/hip_runtime_api.h> are
incompatible with C, so they should only be included in the rocBLAS internal
C++ implemenation, not in the public headers, which must be compatible with C.

EOF
}

# Returns whether the output file is up to date.
# Prints the output file.
# The first argument is the source header file name.
# The second argument is the suffix to use for the output file.
# If the third argument is empty, the return value is always false.
out_uptodate()
{
    local file="$1_$2"
    local out="build/compilation_tests/$file.o"
    mkdir -p $(dirname "$out")
    realpath "$out"
    [[ -n "$3" && "$out" -nt "$script" ]] || return
    find library clients \( -iname \*.hpp -o -iname \*.h \) -print0 \
        | while read -r -d $'\0' file; do
        [[ "$out" -nt "$file" ]] || return
    done
}

HCC=/opt/rocm/hcc/bin/hcc

HCC_OPTS="-Werror -DBUILD_WITH_TENSILE=1 -DTensile_RUNTIME_LANGUAGE_HIP=1 -DTensile_RUNTIME_LANGUAGE_OCL=0 -Drocblas_EXPORTS -I$(realpath library/include) -I$(realpath library/src/include) -I$(realpath build/release/include) -I$(realpath library/src/blas3/Tensile) -isystem /opt/rocm/hip/include -isystem /opt/rocm/hsa/include -isystem /opt/rocm/hcc/include -isystem /opt/rocm/include -I$(realpath build/release/Tensile) -O3 -DNDEBUG -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -Wno-unused-command-line-argument"

GPU_OPTS="-hc -fno-gpu-rdc --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906 -Werror"

CLANG=/opt/rocm/llvm/bin/clang
CLANG_OPTS="-xc-header -std=c99 -D__HIP_PLATFORM_HCC__"

C99="$HCC -xc-header -std=c99"
CPP11="$HCC -xc++-header -std=c++11"
CPP14="$HCC -xc++-header -std=c++14"

# Every header file must compile on its own, by including all of its
# dependencies. This avoids creating dependencies on the order of
# included files. We define _ROCBLAS_INTERNAL_BFLOAT16_ to enable the
# internal rocblas_bfloat16 code. testing_trmm.hpp is excluded for now.
#
find library clients \( -iname \*.hpp -o -iname \*.h \) \
     \! -name testing_trmm.hpp -print0 | while read -r -d $'\0' file; do
    if ! out=$(out_uptodate $file cpp14 true) && \
            ! echocmd $CPP14 -c -o "$out" -D_ROCBLAS_INTERNAL_BFLOAT16_ \
              $HCC_OPTS $GPU_OPTS "$file"; then
        rm -f "$out"
        cat <<EOF

The header file $file cannot be compiled by itself,
probably because of unmet dependencies on other header files.

Add the necessary #include files at the top of $file
so that $file can be used in any context, without
depending on other files being #included before it is #included.

This allows clang-format to reorder #include directives in a canonical order
without breaking dependencies, because every #include file will first #include
the other ones that it depends on, so the order of the #includes in a single
file will not matter.

EOF
        exit 1
    fi
done

# The headers in library/include must compile with clang host, C99 or C++11,
# for client code.
#
for file in library/include/*.{h,in}; do
    if [[ -x "$CLANG" ]]; then
        if ! out=$(out_uptodate $file clang) && \
                ! echocmd $CLANG $CLANG_OPTS -c -o "$out" $HCC_OPTS $file; then
            rm -f "$out"
            cat <<EOF

The public header $file cannot be compiled with clang host-only
compiler. rocBLAS public headers need to be compatible with host-only
compilers.
EOF
            hip_warning
            exit 1
        fi
    fi
    if ! out=$(out_uptodate $file c99) && \
            ! echocmd $C99 -c -o "$out" $HCC_OPTS $GPU_OPTS $file; then
        rm -f "$out"
        cat <<EOF

The public header $file cannot be compiled with a C compiler.
rocBLAS public headers need to be compatible with C99.
EOF
        hip_warning
        exit 1
    elif ! out=$(out_uptodate $file cpp11) && \
            ! echocmd $CPP11 -c -o "$out" $HCC_OPTS $GPU_OPTS $file; then
        rm -f "$out"
        cat <<EOF

The public header $file cannot be compiled with -std=c++11.
rocBLAS public headers need to be compatible with C++11.

EOF
        exit 1
    fi
done

cat <<EOF
-------------------------------------------------------------------------------
All header compilation tests passed.

Public headers can compile with host-only Clang, -std=c99, and -std=c++11.

All public and internal implementation header files can compile on their own,
without depending on #include file order.

EOF
exit 0
