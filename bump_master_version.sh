#!/bin/bash

# This script needs to be edited to bump old staging version to new master version for new release.
# - run this script in staging branch
# - after running this script merge staging into master
# - after running this script and merging staging into master, run bump_staging_version.sh in master and
#   merge master into staging

OLD_TENSILE_VERSION="Tensile 4.33.0"
NEW_TENSILE_VERSION="Tensile 4.34.0"

OLD_TENSILE_HASH="11d18c76e5107452e1523260b3e8e04c7fcee920"
NEW_TENSILE_HASH="f4f267ff97799b76655daf9006cbb84c984551d7"

# OLD_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.6.0"
# NEW_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.7.1"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

#only update yaml files for a Tensile major version change
#for FILE in library/src/blas3/Tensile/Logic/*/*yaml
#do
#  sed -i "s/${OLD_MINIMUM_REQUIRED_VERSION}/${NEW_MINIMUM_REQUIRED_VERSION}/" $FILE
#done
