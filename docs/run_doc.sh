#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sh run_doxygen.sh

#sed -e 's/ROCBLAS_EXPORT //g' ../library/include/rocblas-functions.h > rocblas-functions.h
#sed -e 's/ROCBLAS_EXPORT //g' ../library/include/rocblas-types.h > rocblas-types.h
#sed -e 's/ROCBLAS_EXPORT //g' ../library/include/rocblas-auxilliary.h > rocblas-auxilliary.h


#doxygen Doxyfile

cd source
make clean
make html
cd ..

rm rocblas.h
