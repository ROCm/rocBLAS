# ########################################################################
# A helper function to generate packaging scripts to register libraries with system
# ########################################################################
function( write_rocm_package_script_files scripts_write_dir library_name library_link_name )

set( ld_conf_file "/etc/ld.so.conf.d/${library_name}-dev.conf" )

file( WRITE ${scripts_write_dir}/postinst
"#!/bin/bash

set -e

do_ldconfig() {
  echo ${CPACK_PACKAGING_INSTALL_PREFIX}/${LIB_INSTALL_DIR} > ${ld_conf_file} && ldconfig
}

do_softlinks() {
    ln -sr ${CPACK_PACKAGING_INSTALL_PREFIX}/${INCLUDE_INSTALL_DIR} ${CPACK_PACKAGING_INSTALL_PREFIX}/../include/${library_name}
    ln -sr ${CPACK_PACKAGING_INSTALL_PREFIX}/${LIB_INSTALL_DIR}/${library_link_name}.1 ${CPACK_PACKAGING_INSTALL_PREFIX}/../lib/${library_link_name}
    ln -sr ${CPACK_PACKAGING_INSTALL_PREFIX}/${LIB_INSTALL_DIR}/cmake/${library_name} ${CPACK_PACKAGING_INSTALL_PREFIX}/../lib/cmake/${library_name}
}

case \"\$1\" in
   configure)
        do_ldconfig
        do_softlinks
   ;;
   abort-upgrade|abort-remove|abort-deconfigure)
        echo \"\$1\"
   ;;
   *)
        exit 0
   ;;
esac
" )

file( WRITE ${scripts_write_dir}/prerm
"#!/bin/bash

set -e

rm_ldconfig() {
    rm -f ${ld_conf_file} && ldconfig
}

rm_softlinks() {
    rm -f ${CPACK_PACKAGING_INSTALL_PREFIX}/../include/${library_name}
    rm -f ${CPACK_PACKAGING_INSTALL_PREFIX}/../lib/${library_link_name}
    rm -f ${CPACK_PACKAGING_INSTALL_PREFIX}/../lib/cmake/${library_name}
}

case \"\$1\" in
   remove|purge)
       rm_ldconfig
       rm_softlinks
   ;;
   *)
        exit 0
   ;;
esac
" )

endfunction( )
