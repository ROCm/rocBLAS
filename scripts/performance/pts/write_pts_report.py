"""Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
from os import path
sys.path.append(path.join(path.dirname( path.dirname(path.abspath(__file__))), 'blas'))
import getspecs
import yaml
import subprocess
import math
import os
from collections import defaultdict
import re
from pathlib import Path
from git_info import create_github_file

# Parameters for output csv file
trackedParamList = ['function', 'precision', 'a_type', 'b_type', 'c_type', 'd_type', 'compute_type', 'input_type', 'output_type',
                    'transA', 'transB', 'uplo', 'diag', 'side', 'M', 'N', 'K', 'KL', 'KU', 'alpha', 'alphai', 'beta', 'betai',
                    'incx', 'incy', 'lda', 'ldb', 'ldd', 'stride_x', 'stride_y', 'stride_a', 'stride_b', 'stride_c', 'stride_d',
                    'batch_count', 'algo', 'solution_index', 'flags', 'iters', 'cold_iters', 'pointer_mode',
                    'mean_gflops', 'median_gflops', 'sample_num', 'rocblas-Gflops'] #Outputs

outputIndex = trackedParamList.index('mean_gflops')

exInputs = {'gemm': 'a_type',
            'trsm': 'a_type',
            'axpy': 'a_type'}

exOutputs = {'gemm': 'c_type',
             'trsm': 'a_type',
             'axpy': 'c_type'}

def args2Dict(args):
    if type(args) == dict:
        return args
    elif type(args) == str:
        # Extend to account for arguments with '-' in the middle
        arguments = args.split(' ')
        res =[(key.replace('-', ''),
                      value if key.startswith('-') and not value.startswith('-') else '')
                      for key, value in zip(arguments, arguments[1:])]
        return dict(res)



def getFunctionTypes(yamlArgs):
    print(yamlArgs)
    if '_ex' in yamlArgs['function']:
        function = yamlArgs['function'].split('_')
        return {'input_type': yamlArgs[exInputs[function[0]]],
                'output_type': yamlArgs[exOutputs[function[0]]],
                'compute_type': yamlArgs['compute_type']}
    else:
        return {'input_type': yamlArgs['precision'],
                'output_type': yamlArgs['precision'],
                'compute_type': yamlArgs['precision']}

def updateProblemList(problemList, testType, commitHash, problemDesc, efficiency):
    for test in problemList:
        if test['TestType']   == testType and \
           test['Parameters'] == problemDesc:
            if commitHash in test['Samples'].keys():
                test['Samples'][commitHash] += [efficiency]
            else:
                test['Samples'][commitHash] = [efficiency]
            break
    else:
        problemList += [{'TestType': testType,
                        'Parameters': problemDesc,
                        'Samples': {commitHash: [efficiency]}}]

def getEnvironmentInfo(device_num):
    rv = {}
    host_info = {}
    host_info['hostname'] = getspecs.gethostname()
    host_info['cpu info'] = getspecs.getcpu()
    host_info['ram'] = getspecs.getram()
    host_info['distro'] = getspecs.getdistro()
    host_info['kernel version'] = getspecs.getkernel()
    host_info['rocm version'] = getspecs.getrocmversion()
    rv['Host info'] = host_info

    device_info = {}
    device_info['device'] = getspecs.getdeviceinfo(device_num, False)
    device_info['vbios version'] = getspecs.getvbios(device_num, False)
    device_info['vram'] = getspecs.getvram(device_num, False)
    device_info['performance level'] = getspecs.getperflevel(device_num, False)
    device_info['system clock'] = getspecs.getsclk(device_num, False)
    device_info['memory clock'] = getspecs.getmclk(device_num, False)
    rv['Device info'] = device_info

    return rv

#d is default dict or contains a superset of trackedParams
def extractTrackedParams(d):
    return [d[p] for p in trackedParamList]

def parseRocblasBenchOutput(output, yamlArgs):
    csvKeys = ''

    lineLists = []

    captureNext = False
    yamlArgs = yamlArgs.split('\n')
    for line in output.split('\n'):
        if captureNext:
            #print("CAPTURE=",line)
            #dd = defaultdict(str, args2Dict(yamlArgs[len(lineLists)]).items())
            #dd.update(getFunctionTypes(args2Dict(yamlArgs[len(lineLists)])))
            dd_output = defaultdict(str, zip(csvKeys, line.split(',')))
            #dd.update(dd_output)
            dd = dd_output
            lineLists += [extractTrackedParams(dd)]
            captureNext = False
        elif line.startswith('function'):
            csvKeys = line.split(',')
            #print("KEYS=",csvKeys)
            captureNext = True

    return lineLists

def addSample(csvLists, newSample):
    newList = []
    matched = False
    for row in csvLists:
        if row[:outputIndex] == newSample[:outputIndex] and not matched:
            row += [newSample[-1]]
            row[trackedParamList.index('sample_num')] = len(row) - len(trackedParamList) + 1
            matched = True
        newList += [row]

    if not matched:
        newSample[trackedParamList.index('sample_num')] = len(newSample) - len(trackedParamList) + 1
        newList += [newSample]

    return newList

def calculateMean(row):
    l = [float(v) for v in row[trackedParamList.index('rocblas-Gflops'):]]
    return sum(l)/float(len(l))

def calculateMedian(row):
    l = [float(v) for v in row[trackedParamList.index('rocblas-Gflops'):]]
    l.sort()
    return l[math.floor(len(l)/2)]


def saveRocblasBenchResults(rocblasBenchCommand, problemsYaml, samples, outputFile):

    with open(problemsYaml, 'r') as file:
        problems = file.read()

    csvLists = [trackedParamList]

    for i in range(0, int(samples)):
        print('Sample ({}/{})'.format(i+1, samples))
        output = subprocess.check_output([rocblasBenchCommand,
                                          '--log_function_name',
                                          '--log_datatype',
                                          '--yaml', problemsYaml])

        output = output.decode('utf-8')

        print(output)

        benchResults = parseRocblasBenchOutput(output, problems)

        for sample in benchResults:
            csvLists = addSample(csvLists, sample)

    for problem in csvLists[1:]:
        problem[trackedParamList.index('mean_gflops')] = calculateMean(problem)
        problem[trackedParamList.index('median_gflops')] = calculateMedian(problem)

    content = ''
    for line in csvLists:
        content += ','.join([str(e) for e in line])+'\n'

    with open(outputFile, 'w') as f:
        print("Writing results to {}".format(outputFile))
        f.write(content)

def main(args):
    if len(args) < 4:
        print('Usage:\n\tpython3 write_pts_report.py bench_executable path tag benchfile1 benchfile2...')
        return 0

    rocblasBenchCommand = args[0]
    dirName             = args[1]
    tag                 = args[2]
    filenames           = args[3:]

    for filename in filenames:
        print("==================================\n|| Running benchmarks from {}\n==================================".format(filename))
        filePrefix = Path(filename).stem
        subDirectory = os.path.join(dirName, "rocBLAS_PTS_Benchmarks_"+filePrefix, tag)
        Path(subDirectory).mkdir(parents=True, exist_ok=True)

        outputName, _ = os.path.splitext(os.path.basename(filename))

        saveRocblasBenchResults(rocblasBenchCommand,
                                filename,
                                50,
                                os.path.join(subDirectory, outputName+'_benchmark.csv'))

        environmentInfo = getEnvironmentInfo(0)

        with open(os.path.join(subDirectory, 'specs.txt'), 'w') as f:
            for category, params in environmentInfo.items():
                f.write(category+':\n')
                for key, value in params.items():
                    f.write('    '+str(key)+': '+str(value)+'\n')

        # Will only be correct if script is run from directory of the git repo associated
        # with the rocblas-bench executable
        create_github_file(os.path.join(subDirectory, 'rocBLAS-commit-hash.txt'))

if __name__=='__main__':
    main(sys.argv[1:])
