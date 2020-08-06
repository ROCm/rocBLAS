#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sed -e 's/ROCBLAS_EXPORT //g' ../library/include/rocblas.h > rocblas.h
sed -e 's/ROCBLAS_EXPORT //g' ../library/include/internal/rocblas-functions.h > rocblas-functions.h
sed -e 's/ROCBLAS_EXPORT //g' ../library/include/internal/rocblas-types.h > rocblas-types.h
sed -e 's/ROCBLAS_EXPORT //g' ../library/include/internal/rocblas_bfloat16.h > rocblas_bfloat16.h
sed -e 's/ROCBLAS_EXPORT //g' ../library/include/internal/rocblas-complex-types.h > rocblas-complex-types.h
sed -e 's/ROCBLAS_EXPORT //g' ../library/include/internal/rocblas-auxiliary.h > rocblas-auxiliary.h



doxygen Doxyfile
rm *.h
