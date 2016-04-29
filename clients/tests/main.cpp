// ==================================================
// Copyright 2016 Advanced Micro Devices, Inc.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// @file
/// @brief googletest based unit tester for rocblas
///

#include <iostream>
#include <cblas.h>
#include <gtest/gtest.h>
// #include <boost/program_options.hpp>

#include "hip_runtime.h"
#include "rocblas.h"

// namespace po = boost::program_options;

int main( int argc, char* argv[] )
{
	// // Declare the supported options.
	// po::options_description desc( "rocblas-test command line options" );
	// desc.add_options()
	// 	( "help,h",		"print help message" )
	// 	;
	//
	// //	Parse the command line options, ignore unrecognized options and collect them into a vector of strings
	// po::variables_map vm;
	// po::parsed_options parsed = po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );
	// po::store( parsed, vm );
	// po::notify( vm );
	// std::vector< std::string > to_pass_further = po::collect_unrecognized( parsed.options, po::include_positional );
	//
	// std::cout << std::endl;
	//
	// if( vm.count( "help" ) )
	// {
	// 	std::cout << desc << std::endl;
	// 	return 0;
	// }

	::testing::InitGoogleTest( &argc, argv );

	return RUN_ALL_TESTS();

}
