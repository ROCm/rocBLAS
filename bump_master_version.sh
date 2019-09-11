#!/bin/bash

# This script needs to be edited to bump old develop version to new master version for new release.
# - run this script in develop branch 
# - after running this script merge develop into master
# - after running this script and merging develop into master, run bump_develop_version.sh in master and
#   merge master into develop

OLD_ROCBLAS_VERSION="2.7.0"
NEW_ROCBLAS_VERSION="2.8.0"

OLD_TENSILE_VERSION="tensile_tag 9c63a0bf1c0acdb44376ddc80b867beb3386981a"
NEW_TENSILE_VERSION="tensile_tag 015477ad8c8ea0ef6f59b8d49a65015b46b8a48e"

OLD_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.6.0"
NEW_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.7.1"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

#only update yaml files for a Tensile major version change
#for FILE in library/src/blas3/Tensile/Logic/*/*yaml
#do
#  sed -i "s/${OLD_MINIMUM_REQUIRED_VERSION}/${NEW_MINIMUM_REQUIRED_VERSION}/" $FILE
#done
