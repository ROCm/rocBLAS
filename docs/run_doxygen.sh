#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sed -e 's/ROCBLAS_EXPORT //g' ../library/include/rocblas.h > rocblas.h
doxygen Doxyfile
rm rocblas.h
