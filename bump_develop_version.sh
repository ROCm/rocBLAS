#!/bin/bash

# for the develop branch this script bumps the Tensile version and hash and the rocBLAS version

OLD_TENSILE_VERSION="TENSILE_VERSION 4.38.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.39.0"

OLD_TENSILE_HASH="61d7f30cf3a79d8b02b9f89ce7e07f7d4e90040c"
NEW_TENSILE_HASH="92fff93ae6f413535639d9341511ee9f0b41046f"

OLD_ROCBLAS_VERSION="4.0.0"
NEW_ROCBLAS_VERSION="4.1.0"

OLD_SO_VERSION="rocblas_SOVERSION 4.0"
NEW_SO_VERSION="rocblas_SOVERSION 4.1"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt
