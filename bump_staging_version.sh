#!/bin/bash

# This script needs to be edited to bump new master hash to new staging hash for new release.
# - run this script after running bump_master_version.sh and merging staging into master
# - run this script in master branch
# - after running this script merge master into staging

OLD_TENSILE_HASH="f4f267ff97799b76655daf9006cbb84c984551d7"
NEW_TENSILE_HASH="f4af1648af25e7b36a5e6a17e8465ab402129fb2"

OLD_ROCBLAS_VERSION="2.45.0"
NEW_ROCBLAS_VERSION="2.46.0"

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

