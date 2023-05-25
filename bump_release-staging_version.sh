#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="TENSILE_VERSION 4.37.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.38.0"

OLD_TENSILE_HASH="0884a15d3e0fc2b56d590725c8b7ca8ecc3abbb9"
NEW_TENSILE_HASH="bbc96e67e96f178b1d9400473e020a2a5c6bafb3"

OLD_SO_VERSION="rocblas_SOVERSION 3.1.0"
NEW_SO_VERSION="rocblas_SOVERSION 3.1.0"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt
