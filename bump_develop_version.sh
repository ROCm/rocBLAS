#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.30.0"
NEW_ROCBLAS_VERSION="2.31.0"

OLD_TENSILE_VERSION="Tensile 4.23.0"
NEW_TENSILE_VERSION="Tensile 4.23.0"

OLD_TENSILE_HASH="b68edc65aaeed08c71b2b8622f69f83498b57d7a"
NEW_TENSILE_HASH="f8a03abb87ab897d3e9a005c2b88c5a50d3c5718"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

