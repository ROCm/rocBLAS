#!/bin/bash

# for the develop branch this script bumps the Tensile version and hash and the rocBLAS version

OLD_TENSILE_VERSION="TENSILE_VERSION 4.37.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.38.0"

OLD_TENSILE_HASH="0884a15d3e0fc2b56d590725c8b7ca8ecc3abbb9"
NEW_TENSILE_HASH="38cc25330f6ad7bf8187cd61c2a944ff11a1a921"

OLD_ROCBLAS_VERSION="3.1.0"
NEW_ROCBLAS_VERSION="4.0.0"

OLD_SO_VERSION="rocblas_SOVERSION 3.1"
NEW_SO_VERSION="rocblas_SOVERSION 4.0"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt
