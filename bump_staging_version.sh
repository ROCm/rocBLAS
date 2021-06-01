#!/bin/bash

# This script needs to be edited to bump new master hash to new staging hash for new release.
# - run this script after running bump_master_version.sh and merging staging into master
# - run this script in master branch
# - after running this script merge master into staging

OLD_TENSILE_HASH="9cbabb07f81e932b9c98bf5ae48fbd7fcef615cf"
NEW_TENSILE_HASH="057d389dd902eae437576b505204a22daeb4ae1d"

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

