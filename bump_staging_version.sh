#!/bin/bash

# This script needs to be edited to bump new master hash to new staging hash for new release.
# - run this script after running bump_master_version.sh and merging staging into master
# - run this script in master branch
# - after running this script merge master into staging

OLD_TENSILE_HASH="0f6a6d1557868d6d563cb1edf167c32c2e34fda0"
NEW_TENSILE_HASH="8b64484697853db43e008499a60b38d363199713"

OLD_ROCBLAS_VERSION="2.41.0"
NEW_ROCBLAS_VERSION="2.42.0"

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

