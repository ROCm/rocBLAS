#!/bin/bash

# for the develop branch this script bumps the Tensile version and hash and the rocBLAS version

OLD_TENSILE_VERSION="Tensile 4.36.0"
NEW_TENSILE_VERSION="Tensile 4.37.0"

OLD_TENSILE_HASH="d78dde305d0d3c1754c8e9ce62b3290358dd3f4c"
NEW_TENSILE_HASH="69c03958b9dab1c7211df365c26394ace7ff55f0"

OLD_ROCBLAS_VERSION="3.0.0"
NEW_ROCBLAS_VERSION="3.1.0"

OLD_SO_VERSION="rocblas_SOVERSION 0.1"
NEW_SO_VERSION="rocblas_SOVERSION 3.0"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt
