#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="TENSILE_VERSION 4.37.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.38.0"

OLD_TENSILE_HASH="aba52fa129099cd7c32b322f5daa1a586ad0792b"
NEW_TENSILE_HASH="bbc96e67e96f178b1d9400473e020a2a5c6bafb3"

OLD_SO_VERSION="rocblas_SOVERSION 3.1.0"
NEW_SO_VERSION="rocblas_SOVERSION 3.1.0"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt
