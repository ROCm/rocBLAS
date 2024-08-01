#!/bin/bash

# for the develop branch this script bumps the Tensile version and hash and the rocBLAS version

OLD_TENSILE_VERSION="TENSILE_VERSION 4.41.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.42.0"

OLD_TENSILE_HASH="34a42538e425c330b52694e3adf9fcda537db1dd"
NEW_TENSILE_HASH="2f9ef2fbe7732dd9923f1c7a22a530ba2b651ea2"

OLD_ROCBLAS_VERSION="4.3.0"
NEW_ROCBLAS_VERSION="4.4.0"

OLD_SO_VERSION="rocblas_SOVERSION 4.3"
NEW_SO_VERSION="rocblas_SOVERSION 4.4"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt
