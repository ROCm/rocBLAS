#!/bin/bash

# This script needs to be edited to bump new master hash to new staging hash for new release.
# - run this script after running bump_master_version.sh and merging staging into master
# - run this script in master branch
# - after running this script merge master into staging

OLD_TENSILE_HASH="4ddf088d83d4e287f7bc3fc0631c2c5dba49cf8e"
NEW_TENSILE_HASH="4ddf088d83d4e287f7bc3fc0631c2c5dba49cf8e"

OLD_ROCBLAS_VERSION="2.42.0"
NEW_ROCBLAS_VERSION="2.43.0"

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

