#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.16.0"
NEW_ROCBLAS_VERSION="2.17.0"

OLD_TENSILE_VERSION="Tensile 4.16.0"
NEW_TENSILE_VERSION="Tensile 4.16.0"

OLD_TENSILE_HASH="66487a1e7756bc2a5a9005cea0f903308e63f248"
NEW_TENSILE_HASH="f93c06c9061bf1d9d40453b359879ad1fc29ba67"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

