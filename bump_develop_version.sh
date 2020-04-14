#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.20.0"
NEW_ROCBLAS_VERSION="2.21.0"

OLD_TENSILE_VERSION="Tensile 4.18.0"
NEW_TENSILE_VERSION="Tensile 4.18.0"

OLD_TENSILE_HASH="e3ef43d2b85c33aac5e298ef10ccc8d00d5c602d"
NEW_TENSILE_HASH="62fb9a16909ddef08010915cfefe4c0341f48daa"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

