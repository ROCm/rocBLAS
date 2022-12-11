#!/bin/bash

# This script needs to be edited to bump new master hash to new staging hash for new release.
# - run this script after running bump_master_version.sh and merging staging into master
# - run this script in master branch
# - after running this script merge master into staging

OLD_TENSILE_HASH="3e7f496a82721e55a856212f99502b121e38e2f0"
NEW_TENSILE_HASH="ecfa61ed356371f497bdca99324acedcabfbd99f"

OLD_ROCBLAS_VERSION="2.47.0"
NEW_ROCBLAS_VERSION="2.48.0"

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

