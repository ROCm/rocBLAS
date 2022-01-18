#!/bin/bash

# This script needs to be edited to bump new master hash to new staging hash for new release.
# - run this script after running bump_master_version.sh and merging staging into master
# - run this script in master branch
# - after running this script merge master into staging

OLD_TENSILE_HASH="730f511a39859c7e2911bb0056d7ddada0ec8e58"
NEW_TENSILE_HASH="2b57e4388b40556cf538ad62d4db69e4795853a1"

OLD_ROCBLAS_VERSION="2.43.0"
NEW_ROCBLAS_VERSION="2.44.0"

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

