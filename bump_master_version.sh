#!/bin/bash

# This script needs to be edited to bump old develop version to new master version for new release.
# - run this script in develop branch
# - after running this script merge develop into master
# - after running this script and merging develop into master, run bump_develop_version.sh in master and
#   merge master into develop

OLD_ROCBLAS_VERSION="2.39.0"
NEW_ROCBLAS_VERSION="2.39.0"

OLD_TENSILE_VERSION="Tensile 4.27.0"
NEW_TENSILE_VERSION="Tensile 4.28.0"

OLD_TENSILE_HASH="fa6a88181754c430b81144db4360f3d09b05a05f"
NEW_TENSILE_HASH="9cbabb07f81e932b9c98bf5ae48fbd7fcef615cf"

OLD_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.6.0"
NEW_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.7.1"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

#only update yaml files for a Tensile major version change
#for FILE in library/src/blas3/Tensile/Logic/*/*yaml
#do
#  sed -i "s/${OLD_MINIMUM_REQUIRED_VERSION}/${NEW_MINIMUM_REQUIRED_VERSION}/" $FILE
#a9379f4e42efb754c9a618047bfbf292d74dfd0fdone
