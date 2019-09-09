
# Modified from https://github.com/OPM/opm-common/blob/master/cmake/Modules/UseSystemInfo.cmake based on GNU General Public License v3.0 from https://github.com/OPM/opm-common/blob/master/LICENSE

function (get_os_name OS_NAME)
  set(_os_name)
  file (GLOB open_os_release /etc/os-release)
  if (NOT open_os_release STREQUAL "")
    read_value_from_os_release("ID" _os_name)
  else()
     set( _os_name "unknown" )
   endif()
   set(${OS_NAME} ${_os_name} PARENT_SCOPE)
   set(${OS_NAME}_${_os_name} TRUE PARENT_SCOPE)
 endfunction()

function (read_value_from_os_release KEYVALUE OUTPUT)
  file (STRINGS /etc/os-release _os_name_line
    REGEX "^${KEYVALUE}="
    )

  set(_output)
  string (REGEX REPLACE
    "^${KEYVALUE}=\"?\(.*\)" "\\1" _output "${_os_name_line}"
    )

  #remove tailing quote
  string (REGEX REPLACE
    "\"$" "" _output "${_output}"
    )
  set(${OUTPUT} ${_output} PARENT_SCOPE)
endfunction ()
