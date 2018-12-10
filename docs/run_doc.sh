#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sed -e 's/ROCBLAS_EXPORT //g' ../library/include/rocblas-functions.h > rocblas.h
doxygen Doxyfile

cd source
make clean
make html
cd ..

rm rocblas.h
