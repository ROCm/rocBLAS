# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Visual studio based builds generate binary bitness based upon the selected generator.
# For GNU or CLANG based compilers, binary bitness is based on compiler flags
if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
  string( REGEX MATCH "-m(64|32)" mflag_cxx "${CMAKE_CXX_FLAGS}" )
  string( REGEX MATCH "-m(64|32)" mflag_c "${CMAKE_C_FLAGS}" )

	# If user specified no bitness flags at configure time, default to 64-bit
  if( NOT( mflag_cxx OR mflag_c ) )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64" )
		set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64" )
  endif( )

	# Explicitely check if user wants 32-bit libraries
	string( REGEX MATCH "-m64" mflag_cxx_x64 "${CMAKE_CXX_FLAGS}" )
  string( REGEX MATCH "-m64" mflag_c_x64 "${CMAKE_C_FLAGS}" )

	if( mflag_cxx_x64 OR mflag_c_x64 )
    set( BUILD_64 ON )
	else( )
		set( BUILD_64 OFF )
  endif( )
  # message( STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  # message( STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
elseif( MSVC_IDE )
	set( BUILD_64 ${CMAKE_CL_64} )
	set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
endif( )
