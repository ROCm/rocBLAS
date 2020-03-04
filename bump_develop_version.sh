#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.18.0"
NEW_ROCBLAS_VERSION="2.19.0"

OLD_TENSILE_VERSION="Tensile 4.17.0"
NEW_TENSILE_VERSION="Tensile 4.17.0"

OLD_TENSILE_HASH="fdd9ef8d5a0687596efee85b7ec187f1fb097087"
NEW_TENSILE_HASH="5da063e51737cab9596726199988e5b8e89d9469"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

