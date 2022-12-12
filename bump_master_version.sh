#!/bin/bash

# This script needs to be edited to bump old staging version to new master version for new release.
# - run this script in staging branch
# - after running this script merge staging into master
# - after running this script and merging staging into master, run bump_staging_version.sh in master and
#   merge master into staging

OLD_TENSILE_VERSION="Tensile 4.35.0"
NEW_TENSILE_VERSION="Tensile 4.36.0"

OLD_TENSILE_HASH="db57a7a16be3544cfae3a525f80c0d78dfb24399"
NEW_TENSILE_HASH="3e7f496a82721e55a856212f99502b121e38e2f0"

# OLD_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.6.0"
# NEW_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.7.1"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

#only update yaml files for a Tensile major version change
#for FILE in library/src/blas3/Tensile/Logic/*/*yaml
#do
#  sed -i "s/${OLD_MINIMUM_REQUIRED_VERSION}/${NEW_MINIMUM_REQUIRED_VERSION}/" $FILE
#done
