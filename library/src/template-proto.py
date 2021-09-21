# Copyright 2020-2021 Advanced Micro Devices, Inc.

import os
import argparse
import glob
import re

###############################################################################
# Template to prototype translator
###############################################################################

gPattern = re.compile(r'\{[^\{\}]*\}')


def translateToProto(templateCode):
    global gPattern
    proto = ''.join(templateCode)
    if re.search("ROCBLAS_INTERNAL_EXPORT_NOINLINE", proto) is None:
        return
    # keep warning in proto
    proto = re.sub("ROCBLAS_INTERNAL_EXPORT_NOINLINE", "ROCBLAS_INTERNAL_DEPRECATION", proto)
    n = 1
    while n:
        proto, n = re.subn(gPattern, '', proto)

    if (proto.rstrip()).endswith(';'):
        print(proto.rstrip() + "\n")
    else:
        print(proto.rstrip() + ";\n")


def parseForExportedTemplates(inputFileName):
    with open(inputFileName) as f:
        haveTemplate = False
        lines = f.readlines()
        for line in lines:
            filter = re.match(r'^template', line)
            if (filter):
                if (haveTemplate):
                    translateToProto(body)
                haveTemplate = True
                body = []
                body.append(line)
            elif (haveTemplate):
                body.append(line)
                if re.match(r'^\}', line) is not None:
                    translateToProto(body)
                    haveTemplate = False
        if (haveTemplate):
            translateToProto(body)


def RunExporter():
    print("""
// Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.

// Script-generated file -- do not edit

// rocBLAS internal API may change each release. The rocBLAS team strongly advises against its use.

#pragma once

#include "internal/rocblas-types.h"

""")

    # Parse Command Line Arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('path', nargs='+',
                           help='Path of a files or directory')
    args = argParser.parse_args()

    # Parse paths
    full_paths = [os.path.join(os.getcwd(), path) for path in args.path]
    files = set()
    for path in full_paths:
        if os.path.isfile(path):
            files.add(path)
        else:
            files.update(glob.glob(path+'*.h*'))

    headerFiles = sorted(files)
    for f in headerFiles:
        parseForExportedTemplates(f)


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    RunExporter()
