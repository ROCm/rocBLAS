#!/usr/bin/python3
"""Copyright 2020-2021 Advanced Micro Devices, Inc.
Manage build and installation"""

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib
from fnmatch import fnmatchcase

args = {}
param = {}
OS_info = {}

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""Checks build arguments""")
    # common
    parser.add_argument('-g', '--debug', required=False, default = False,  action='store_true',
                        help='Generate Debug build (optional, default: False)')
    parser.add_argument(      '--build_dir', type=str, required=False, default = "build",
                        help='Build directory path (optional, default: build)')
    parser.add_argument(      '--skip_ld_conf_entry', required=False, default = False)
    parser.add_argument(      '--static', required=False, default = False, dest='static_lib', action='store_true',
                        help='Generate static library build (optional, default: False)')
    parser.add_argument('-c', '--clients', required=False, default = False, dest='build_clients', action='store_true',
                        help='Generate all client builds (optional, default: False)')
    parser.add_argument('-i', '--install', required=False, default = False, dest='install', action='store_true',
                        help='Install after build (optional, default: False)')
    parser.add_argument(      '--cmake-darg', required=False, dest='cmake_dargs', action='append', default=[],
                        help='List of additional cmake defines for builds (optional, e.g. CMAKE)')
    parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
                        help='Verbose build (optional, default: False)')
    # rocblas
    parser.add_argument(     '--clients-only', dest='clients_only', required=False, default = False, action='store_true',
                        help='Build only clients with a pre-built library')
    parser.add_argument(     '--library-path', dest='library_dir_installed', type=str, required=False, default = "", 
                        help='When only building clients, the path to the pre-built rocBLAS library (optional, default: /opt/rocm/rocblas)')
    parser.add_argument(      '--cpu_ref_lib', type=str, required=False, default = "blis",
                        help='Specify library to use for CPU reference code in testing (blis or lapack)')
    parser.add_argument(      '--rocm_dev', type=str, required=False, default = "",
                        help='Set specific rocm-dev version')
    #tensile
    parser.add_argument('-n', '--no_tensile', dest='build_tensile', required=False, default=True, action='store_false')
    parser.add_argument('-a', '--architecture', dest='gpu_architecture', required=False, default="gfx906", #:sramecc+:xnack-" ) #gfx1030" ) #gfx906" ) # gfx1030" )
                        help='Set GPU architectures, e.g. all, gfx000, gfx803, gfx906:xnack-;gfx1030 (optional, default: all)')
    parser.add_argument('-l', '--logic', dest='tensile_logic', type=str, required=False, default="asm_full",
                        help='Set Tensile logic target, e.g., asm_full, asm_lite, etc. (optional, default: asm_full)')
    parser.add_argument('-f', '--fork', dest='tensile_fork', type=str, required=False, default="",
                        help='GitHub Tensile fork to use, e.g., ROCmSoftwarePlatform or MyUserName')
    parser.add_argument('-b', '--branch', dest='tensile_tag', type=str, required=False, default="",
                        help='GitHub Tensile branch or tag to use, e.g., develop, mybranch or <commit hash>')
    parser.add_argument('-t', '--test_local_path', dest='tensile_test_local_path', type=str, required=False, default="",
                        help='Use a local path for Tensile instead of remote GIT repo (optional)')
    parser.add_argument('-u', '--use-custom-version', dest='tensile_version', type=str, required=False, default="",
                        help='Ignore Tensile version and just use the Tensile tag (optional)')
    parser.add_argument(     '--no-merge-files', dest='merge_files', required=False, default=True, action='store_false',
                        help='To disable Tensile_MERGE_FILES (optional)')
    parser.add_argument(     '--no-msgpack', dest='tensile_msgpack_backend', required=False, default=True, action='store_false',
                        help='Set Tensile backend to not use MessagePack and so use YAML (optional)')

    return parser.parse_args()

def os_detect():
    global OS_info
    if os.name == "nt":
        OS_info["ID"] = platform.system()
    else:
        inf_file = "/etc/os-release"
        if os.path.exists(inf_file):
            with open(inf_file) as f:
                for line in f:
                    if "=" in line:
                        k,v = line.strip().split("=")
                        OS_info[k] = v.replace('"','')
    OS_info["NUM_PROC"] = os.cpu_count()
    print(OS_info)

def create_dir(dir_path):
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        run_cmd( "rm" , f"-rf {dir_path}")

def cmake_path(os_path):
    if os.name == "nt":
        return os_path.replace("\\", "/")
    else:
        return os_path
    
def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    cmake_executable = ""
    cmake_options = []
    src_path = cmake_path(cwd_path)
    cmake_platform_opts = []
    if os.name == "nt":
        # not really rocm path as none exist, HIP_DIR set in toolchain is more important
        rocm_path = os.getenv( 'ROCM_CMAKE_PATH', "C:/github/rocm-cmake-master/share/rocm")
        cmake_executable = "cmake"
        #set CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
        cmake_platform_opts.append( f"-DCPACK_PACKAGING_INSTALL_PREFIX=" )
        cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"C:/hipSDK\"" )
        generator = f"-G Ninja"
        cmake_options.append( generator )
        toolchain = os.path.join( src_path, "toolchain-windows.cmake" )
    else:
        rocm_path = os.getenv( 'ROCM_PATH', "/opt/rocm")
        if (OS_info["ID"] in ['centos', 'rhel']):
          cmake_executable = "cmake" # was cmake3 but now we built cmake
        else:
          cmake_executable = "cmake"
        cmake_platform_opts.append( f"-DROCM_DIR:PATH={rocm_path} -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}" )
        cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"rocblas-install\"" )
        toolchain = "toolchain-linux.cmake"

    print( f"Build source path: {src_path}")

    tools = f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
    cmake_options.append( tools )

    cmake_options.extend( cmake_platform_opts )

    cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path}" 
    cmake_options.append( cmake_base_options )

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF" 
    cmake_options.append( cmake_pack_options )

    if os.getenv('CMAKE_CXX_COMPILER_LAUNCHER'):
        cmake_options.append( f"-DCMAKE_CXX_COMPILER_LAUNCHER={os.getenv('CMAKE_CXX_COMPILER_LAUNCHER')}" )

    cmake_options.append("-DBUILD_TESTING=OFF")

    print( cmake_options )

    # build type
    cmake_config = ""
    build_dir = os.path.abspath(args.build_dir)
    if not args.debug:
        build_path = os.path.join(build_dir, "release")
        cmake_config="Release"
    else:
        build_path = os.path.join(build_dir, "debug")
        cmake_config="Debug"

    cmake_options.append( f"-DCMAKE_BUILD_TYPE={cmake_config}" ) 

    # clean
    delete_dir( build_path )

    create_dir( os.path.join(build_path, "clients") )
    os.chdir( build_path )

    if args.static_lib:
        cmake_options.append( f"-DBUILD_SHARED_LIBS=OFF" )

    if args.skip_ld_conf_entry:
        cmake_options.append( f"-DROCM_DISABLE_LDCONFIG=ON" )

    if args.build_clients:
        cmake_build_dir = cmake_path(build_dir)
        cmake_options.append( f"-DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_DIR={cmake_build_dir}" )

    if args.clients_only:
        if args.library_dir_installed:
            library_dir = args.library_dir_installed
        else:
            library_dir = f"{rocm_path}/rocblas"
        cmake_lib_dir = cmake_path(library_dir)
        cmake_options.append( f"-DSKIP_LIBRARY=ON -DROCBLAS_LIBRARY_DIR={cmake_lib_dir}" )

    if args.cpu_ref_lib == 'blis':
        cmake_options.append( f"-DLINK_BLIS=ON" )

    # not just for tensile 
    cmake_options.append( f"-DAMDGPU_TARGETS={args.gpu_architecture}" )

    if not args.build_tensile:
        cmake_options.append( f"-DBUILD_WITH_TENSILE=OFF" )
    else:
        cmake_options.append( f"-DTensile_CODE_OBJECT_VERSION=V3" )
        if args.tensile_logic:
            cmake_options.append( f"-DTensile_LOGIC={args.tensile_logic}" )
        if args.tensile_fork:
            cmake_options.append( f"-Dtensile_fork={args.tensile_fork}" )
        if args.tensile_tag:
            cmake_options.append( f"-Dtensile_tag={args.tensile_tag}" )
        if args.tensile_test_local_path:
            cmake_options.append( f"-DTensile_TEST_LOCAL_PATH={args.tensile_test_local_path}" )
        if args.tensile_version:
            cmake_options.append( f"-DTENSILE_VERSION={args.tensile_version}" )
        if not args.merge_files:
            cmake_options.append( f"-DTensile_MERGE_FILES=OFF" )
        if args.tensile_msgpack_backend:
            cmake_options.append( f"-DTensile_LIBRARY_FORMAT=msgpack" )
        else:
            cmake_options.append( f"-DTensile_LIBRARY_FORMAT=yaml" )

    if args.cmake_dargs:
        for i in args.cmake_dargs:
          cmake_options.append( f"-D{i}" )

    cmake_options.append( f"{src_path}")
    cmd_opts = " ".join(cmake_options)

    return cmake_executable, cmd_opts


def make_cmd():
    global args
    global OS_info

    make_options = []

    nproc = OS_info["NUM_PROC"]
    if os.name == "nt":
        make_executable = f"cmake.exe --build . " # ninja
        if args.verbose:
          make_options.append( "--verbose" )
        make_options.append( "--target all" )
        if args.install:
          make_options.append( "--target package --target install" )
    else:
        make_executable = f"make -j{nproc}"
        if args.verbose:
          make_options.append( "VERBOSE=1" )
        if True: # args.install:
         make_options.append( "install" )
    cmd_opts = " ".join(make_options)

    return make_executable, cmd_opts

def run_cmd(exe, opts):
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode

def main():
    global args
    os_detect()
    args = parse_args()

    # configure
    exe, opts = config_cmd()
    run_cmd(exe, opts)

    # make
    exe, opts = make_cmd()
    run_cmd(exe, opts)

if __name__ == '__main__':
    main()

