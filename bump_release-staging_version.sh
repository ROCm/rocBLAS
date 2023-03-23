#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="Tensile 4.36.0"
NEW_TENSILE_VERSION="Tensile 4.37.0"

OLD_TENSILE_HASH="d78dde305d0d3c1754c8e9ce62b3290358dd3f4c"
NEW_TENSILE_HASH="a15ca875712d0493fab1b1ae04b3ffd882b14937"

OLD_SO_VERSION="rocblas_SOVERSION 0.1"
NEW_SO_VERSION="rocblas_SOVERSION 3.0"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt
