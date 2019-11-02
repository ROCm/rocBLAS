#!/bin/bash

export LOCALE=C
set -e
exec 2>&1

script=$(realpath "$0")

build_first()
{
    echo "Please run this script after at least one build of rocBLAS."
    exit 1
}

[[ ! -e CMakeCache.txt ]] && build_first

BUILD_DIR=$(realpath "$(pwd)")
SOURCE_DIR=$(realpath -m "$(grep CMAKE_HOME_DIRECTORY CMakeCache.txt | sed 's/CMAKE_HOME_DIRECTORY:INTERNAL=//g')")

[[ ! -e $BUILD_DIR/include/rocblas-export.h ]] && build_first

# Returns whether the output file is up to date.
# Prints the output file.
# The first argument is the source header file name.
# The second argument is the suffix to use for the output file.
# If the third argument is empty, the return value is always false.
out_uptodate()
{
    local file="$1_$2"
    local filename="$BUILD_DIR/compilation_tests/$file.o"
    mkdir -p $(dirname "$filename")
    local out=$(realpath -m "$filename")
    echo "$out"
    [[ -n "$3" && "$out" -nt "$script" ]] || return
    find library clients \( -iname \*.hpp -o -iname \*.h \) -print0 \
        | while read -r -d $'\0' file; do
        [[ "$out" -nt "$file" ]] || return
    done
}

HCC=/opt/rocm/hcc/bin/hcc

HCC_OPTS="-Werror -DBUILD_WITH_TENSILE=1 -DTensile_RUNTIME_LANGUAGE_HIP=1 -DTensile_RUNTIME_LANGUAGE_OCL=0 -Drocblas_EXPORTS -I$(realpath library/include) -I$(realpath library/src/include) -I$(realpath $BUILD_DIR/include) -I$(realpath $SOURCE_DIR/library/src/blas3/Tensile) -isystem /opt/rocm/hip/include -isystem /opt/rocm/hsa/include -isystem /opt/rocm/hcc/include -isystem /opt/rocm/include -I$(realpath $BUILD_DIR/Tensile) -O3 -DNDEBUG -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -Wno-unused-command-line-argument"

GPU_OPTS="-hc -fno-gpu-rdc --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906 -Werror"

CLANG=/opt/rocm/llvm/bin/clang
CLANG_OPTS="-xc-header -std=c99"  # auto set in hip_common.h -D__HIP_PLATFORM_HCC__

C99="$HCC -xc-header -std=c99"
CPP11="$HCC -xc++-header -std=c++11"
CPP14="$HCC -xc++-header -std=c++14"

if [[ -e /.dockerenv ]]; then
    NP=4   # limit parallelism to 4
else
    NP=0   # unlimited
fi

# xargs commands to perform parallel builds
xargs_coproc()
{
    { coproc { xargs "-P$NP" -d "\n" -n1 /bin/bash -xc --; } 4>&1 >&3 2>&1; } 3>&1
    XARGS_PID=$!
    exec {XARGS_OUT}<&${COPROC[0]}- {XARGS_IN}>&${COPROC[1]}-
    echo true >&$XARGS_IN  # At least one command is necessary
}

xargs_wait()
{
    XARGS_OUTPUT=""
    exec {XARGS_IN}<&-
    if ! wait $XARGS_PID; then
        read -t 0.1 -u $XARGS_OUT XARGS_OUTPUT
        return 1
    fi
}

# Every header file must compile on its own, by including all of its
# dependencies. This avoids creating dependencies on the order of
# included files. testing_trmm.hpp is excluded for now.
#
xargs_coproc
find library clients \( -iname \*.hpp -o -iname \*.h \) \
     \! -name testing_trmm.hpp -print0 | while read -r -d $'\0' file; do
    out=$(out_uptodate "$file" cpp14 true) || \
        echo "$CPP14 -c -o "$out" $HCC_OPTS $GPU_OPTS "$file" || (rm -f "$out"; echo "$file" >&4; exit 255)" >&$XARGS_IN
done

if ! xargs_wait; then
        cat <<EOF

The header file $XARGS_OUTPUT cannot be compiled by itself,
probably because of unmet dependencies on other header files.

Add the necessary #include files at the top of $XARGS_OUTPUT
so that $XARGS_OUTPUT can be used in any context, without
depending on other files being #included before it is #included.

This allows clang-format to reorder #include directives in a canonical order
without breaking dependencies, because every #include file will first #include
the other ones that it depends on, so the order of the #includes in a single
file will not matter.

EOF
        exit 1
fi

# The headers in $SOURCE_DIR/library/include must compile with clang host, C99 or C++11,
# for client code.
#
if [[ -x "$CLANG" ]]; then
    xargs_coproc
    for file in $SOURCE_DIR/library/include/*.{h,in}; do
        out=$(out_uptodate $file clang) || \
             echo "$CLANG $CLANG_OPTS -c -o "$out" $HCC_OPTS "$file" || (rm -f "$out"; echo "$file" >&4; exit 255)" >&$XARGS_IN
    done

    if ! xargs_wait; then
        cat <<EOF

The public header file $XARGS_OUTPUT cannot be compiled with
clang host-only compiler. rocBLAS public header files need to be compatible
with host-only compilers.

<hip/hip_runtime.h> (and, sometimes due to bugs, <hip/hip_runtime_api.h>) are
incompatible with C, so they should only be included in the rocBLAS internal
C++ implemenation, not in the public headers, which must be compatible with C.
EOF
        exit 1
    fi
fi

xargs_coproc
for file in $SOURCE_DIR/library/include/*.{h,in}; do
    out=$(out_uptodate $file c99) || \
        echo "$C99 -c -o "$out" $HCC_OPTS $GPU_OPTS "$file" || (rm -f "$out"; echo "$file" >&4; exit 255)" >&$XARGS_IN
done

if ! xargs_wait; then
    cat <<EOF

The public header file $XARGS_OUTPUT cannot be compiled with a C compiler.
rocBLAS public headers need to be compatible with C99.

<hip/hip_runtime.h> and (sometimes due to bugs) <hip/hip_runtime_api.h> are
incompatible with C, so they should only be included in the rocBLAS internal
C++ implemenation, not in the public headers, which must be compatible with C.
EOF
        exit 1
fi

xargs_coproc
for file in $SOURCE_DIR/library/include/*.{h,in}; do
    out=$(out_uptodate $file cpp11) ||
        echo "$CPP11 -c -o "$out" $HCC_OPTS $GPU_OPTS "$file" || (rm -f "$out"; echo "$file" >&4; exit 255)" >&$XARGS_IN
done

if ! xargs_wait; then
        cat <<EOF

The public header file $XARGS_OUTPUT cannot be compiled with
-std=c++11. rocBLAS public headers need to be compatible with C++11.

EOF
        exit 1
fi

cat <<EOF
-------------------------------------------------------------------------------
All header file compilation tests passed.

Public header files can compile with host-only Clang, -std=c99, and -std=c++11.

All public and internal implementation header files can compile on their own,
without depending on #include file order.

EOF
