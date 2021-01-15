#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.36.0"
NEW_ROCBLAS_VERSION="2.37.0"

OLD_TENSILE_VERSION="Tensile 4.26.0"
NEW_TENSILE_VERSION="Tensile 4.26.0"

OLD_TENSILE_HASH="d175277084d3253401583aa030aba121e8875bfd"
NEW_TENSILE_HASH="237e7ef88cf7e17378ad6dacfca75f40ee7add86"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

