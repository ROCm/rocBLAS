# Copyright 2020 Advanced Micro Devices, Inc.

import os
import sys
import argparse
import glob
import re

################################################################################
# Template to prototype translator
################################################################################

gPattern = re.compile(r'\{[^\{\}]*\}')

def translateToProto(templateCode):
    global gPattern
    proto = ''.join(templateCode)
    if (re.search("ROCBLAS_EXPORT_NOINLINE", proto) == None):
         return
    proto = re.sub("ROCBLAS_EXPORT_NOINLINE ","", proto )
    n = 1
    while n:
        proto, n = re.subn(gPattern, '', proto)
    print( proto.rstrip() + ";\n" )


def parseForExportedTemplates(inputFileName):
    with open(inputFileName) as f:
        haveTemplate = False
        lines = f.readlines()
        for line in lines:
            filter = re.match(r'^template',line)
            if (filter):
                haveTemplate = True
                body = []
            if (haveTemplate):
                body.append(line)
                if (re.match(r'^\}',line) != None):
                    translateToProto(body)
                    haveTemplate = False
        if (haveTemplate):
            translateToProto(body)


def RunExporter():

    print("")
    print("// Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.")
    print("")
    print("// script generated file do not edit")
    print("")
    print("#pragma once")
    print("")

    # Parse Command Line Arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('path', nargs='+', help='Path of a files or directory')
    args = argParser.parse_args()

    # Parse paths
    full_paths = [os.path.join(os.getcwd(), path) for path in args.path]
    files = set()
    for path in full_paths:
        if os.path.isfile(path):
            files.add(path)
        else:
            files |= set(glob.glob(path + '/*.h*'))

    headerFiles = sorted(files)
    for f in headerFiles:
        parseForExportedTemplates(f)


################################################################################
# Main
################################################################################
if __name__ == "__main__":
    RunExporter()
