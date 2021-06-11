#!/bin/bash

# This script needs to be edited to bump new master hash to new staging hash for new release.
# - run this script after running bump_master_version.sh and merging staging into master
# - run this script in master branch
# - after running this script merge master into staging

OLD_TENSILE_HASH="17aabd14cb7ede34307c40fc87a5074832acf16b"
NEW_TENSILE_HASH="da34e4f61107a9a288ff61e7cf3127f26d242eb4"

OLD_ROCBLAS_VERSION="2.40.0"
NEW_ROCBLAS_VERSION="2.41.0"

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

