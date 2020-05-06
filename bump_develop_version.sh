#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.22.0"
NEW_ROCBLAS_VERSION="2.23.0"

OLD_TENSILE_VERSION="Tensile 4.19.0"
NEW_TENSILE_VERSION="Tensile 4.19.0"

OLD_TENSILE_HASH="bac1ba88980f27ad796fb70f4dd45d4c27235123"
NEW_TENSILE_HASH="fd6df9c22efb21c46f8bef4535d27624abc7ff3a"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

