#!/usr/bin/python3
"""Copyright 2021 Advanced Micro Devices, Inc.
Run tests on build"""

import re
import os
import subprocess
import argparse
import pathlib
import platform
from genericpath import exists
from fnmatch import fnmatchcase


args = {}
param = {}
OS_info = {}
OS_env = {}

TestScript = [ 'cd %IDIR%', 'rocblas-test.exe' ] # --gtest_output=%ODIR%' ]

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""
    Checks build arguments
    """)
    parser.add_argument('-g', '--debug', required=False, default = False,  action='store_true',
                        help='Test Debug build (optional, default: false)')
    parser.add_argument('-o', '--output', type=str, required=False, default = "xml", 
                        help='Test output file (optional, default: test_detail.xml)')
    parser.add_argument(      '--install_dir', type=str, required=False, default = "build", 
                        help='Installation directory where build or release folders are (optional, default: build)')
    # parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
    #                     help='Verbose install (optional, default: False)')
    return parser.parse_args()

def os_detect():
    global OS_env
    global OS_info
    OS_env = os.environ.copy()
    if os.name == "nt":
        OS_info["ID"] = platform.system()
        blis_dir = OS_env["BLIS_DIR"] + "\\lib"
        lapack_dir = OS_env["LAPACK_DIR"] + "\\bin"
        OS_env["PATH"] =  OS_env["PATH"] + ':' + blis_dir + ':' + lapack_dir
        print(OS_env["PATH"])
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
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    return pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        return run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        linux_path = pathlib.Path(dir_path).absolute()
        return run_cmd( "rm" , f"-rf {linux_path}")

def run_cmd(cmd):
    global args
    global OS_env
    if (cmd.startswith('cd ')):
        return os.chdir(cmd[3:])
    if (cmd.startswith('mkdir ')):
        return create_dir(cmd[6:])
    cmdline = f"{cmd}"
    print(cmdline)
    proc = subprocess.Popen(cmdline, stderr=subprocess.STDOUT, env=OS_env)
    status = proc.poll()
    return proc.returncode

def batch(script):
    # 
    cwd = os.curdir
    for i in range(len(script)):
        cmdline = script[i]
        if args.debug: build_type = "debug"
        else: build_type = "release"
        test_dir = f"{args.install_dir}//{build_type}//clients//staging"
        xcmd = cmdline.replace('%IDIR%', test_dir)
        cmd = xcmd.replace('%ODIR%', args.output)
        if cmd.startswith('tdir '):
            if pathlib.Path(cmd[5:]).exists():
                return 0 # all further cmds skipped
            else:
                continue
        error = run_cmd(cmd)
        if (error):
            print(f"ERROR running: {cmd}")
            if (os.curdir != cwd):
                os.chdir( cwd )
            return error
    if (os.curdir != cwd):
        os.chdir( cwd )
    return 0

def run_tests():
    global TestScript
    # install
    cwd = os.curdir

    scripts = []
    scripts.append( TestScript )
    for i in scripts:
        if (batch(i)):
            print("ERROR in script. ABORTING")
            if (os.curdir != cwd):
                os.chdir( cwd )
            return 1       
    if (os.curdir != cwd):
        os.chdir( cwd )
    return 0

def main():
    global args

    os_detect()
    args = parse_args()

    run_tests()

if __name__ == '__main__':
    main()
