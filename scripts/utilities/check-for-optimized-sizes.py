#!/usr/bin/python

import yaml
import sys
import os
import subprocess
from subprocess import PIPE

minimumVersionIdx       = 0
prefixIdx               = 1
architectureIdx         = 2
deviceNamesIdx          = 3
problemTypeStateIdx     = 4
solutionListIdx         = 5
indexOrderIdx           = 6
exactLogicListIdx       = 7       #[[m, n, batch_zie, k, ...], [kernelIdx, GFLOPS]]
mIdx        = 0
nIdx        = 1
batchIdx    = 2
kIdx        = 3
rangeLogicIdx           = 8
tileSelectionLogicIdx   = 9       #Only if tile aware selection is enabled

helpMessage = """Script for determining if a problem size has been benchmarked for.
Usage: 
python3 check-for-optimized-sizes.py -f logfilePath 
                                      [-a architecture] [-c] [-u] [-i] [--help]
Options:
-f       Log file containing benchmark tests.
-a       Architecture. Matches any problem containing this argument. 
         ex: vega10
-c       Deletes cloned directory after completion.
-u       Searches most recent commit for the problem sizes as well.
-i       Use rocBLAS-internal instead of rocBLAS
--help   Displays this message.
"""

usageMessage = """Usage: 
python3 check-for-optimized-sizes.py -f logfilePath 
                                      [-a architecture] [-c] [-u] [-i] [--help]"""


def removeChar(text, removedChar):
    return "".join(filter(lambda c : c != removedChar, text))

class ParseOptionError(Exception):
    pass

#Same usage as getopt but doesn't fail on unknown args and returns dict
def parseOptions(textList, shortArgs, longArgs=[]):
    hasArg = []
    lastChar = ':'
    optionList = []
    #Set keys for short arguments
    for c in shortArgs:
        #':' indicates c takes an argument
        if c == ':':
            if not lastChar.isalpha(): 
                raise ParseOptionError("Error: Colon must follow alphabetic character")
            hasArg[-1] = True
        else:
            optionList.append('-' + c)
            hasArg.append(False)
        lastChar = c

    #Set keys for long arguments
    for s in longArgs:
        #'=' indicates s takes an argument
        if s[-1] == '=':
            hasArg.append(True)
            optionList.append("--" + s[:-1])
        else:
            hasArg.append(False)
            optionList.append("--" + s)

    hasArg = dict(zip(optionList, 
        hasArg))

    #Extract options from textList
    optionDict = dict([])
    needsArg = None 
    for item in textList:
        #If this is an argument of a previous option...
        if needsArg != None:
            #Store associated with it
            optionDict[needsArg] = item
            needsArg = None   
        #Otherwise check if it's a valid option
        elif item in optionList:
            optionDict[item] = None     #Set as none for now
            if hasArg[item]:
                needsArg = item         #Set to capture next item

    return dict(zip(
        list(map(lambda s : removeChar(s, '-'), optionDict.keys())),
        optionDict.values()))

#assumes b has at least as many periods as a
def isVersionGreaterThanOrEqual(versionA, versionB):
    for a, b in zip(versionA.split('.'), versionB.split('.')):
        if int(a) != int(b):
            return int(a) > int(b)
    return True     #Is equal


def shellCmd(command):
    spReturn = subprocess.run(command, shell=True, stdout=PIPE, stderr=PIPE)
    return spReturn.stdout.decode('utf-8').rstrip('\n')


def getInstalledRocBLASCommitHash():
    versionString = shellCmd("dpkg -s rocblas | grep Version") #Format "Version X.XX.X.XXXX-CommitHash"
    if len(versionString) > 0:
        return versionString.split()[1].split('-')[1]
    else:
        return ""


def cloneRepository(destinationPath, cloneInternal):
    print("Cloning repository...")
    if cloneInternal:
        shellCmd("git clone https://github.com/ROCmSoftwarePlatform/rocBLAS-internal.git %s" % destinationPath)
    else:
        shellCmd("git clone https://github.com/ROCmSoftwarePlatform/rocBLAS.git %s" % destinationPath)


def checkoutInstalledBranch(destinationPath):
    rocBLASCommitHash = getInstalledRocBLASCommitHash()
    originalPath = shellCmd("pwd").rstrip('\n')
    os.chdir(destinationPath)
    shellCmd("git checkout %s" % rocBLASCommitHash)
    os.chdir(originalPath)

def checkoutMostRecentBranch(destinationPath): 
    originalPath = shellCmd("pwd").rstrip('\n')
    os.chdir(destinationPath)
    shellCmd("git checkout develop")
    os.chdir(originalPath)


class ProblemDescription:
    def __init__(self, benchmarkText):
        self.m = 1
        self.n = 1
        self.k = 1
        self.batch_count = 1
 
        try:
            optDict= parseOptions(benchmarkText.split(), "m:n:k:", ["batch_count=", "transposeA=", "transposeB="])
            self.m           = int(optDict['m'])
            self.n           = int(optDict['n'])
            self.k           = int(optDict['k'])
            transposeA       =     optDict['transposeA'] == 'T'
            transposeB       =     optDict['transposeB'] == 'T'

        except ValueError:
            print("Error: Problem description parameters have invalid type")
        except ParseOptionError:
            print("Error: Input arguments ill formed")

        if 'batch_count' in optDict.keys():
            self.batch_count = int(optDict['batch_count'])

        self.matrixA = 'Alik' if transposeA else 'Ailk'
        self.matrixB = 'Bjlk' if transposeA else 'Bljk'

    def __str__(self):
        return "(m=%d n=%d k=%d batch_count = %d transA = %r transB = %r)" \
        % (self.m, self.n, self.k, self.batch_count, self.matrixA == 'Alik', self.matrixB == 'Bljk')

def loadBenchmarkDescriptions(logfilePath):
    f = open(logfilePath)
    if f.closed:
        return []

    lines = f.read().split('\n')
    benchmarkList = []

    for line in lines:
        benchmarkList.append(ProblemDescription(line))
        b = benchmarkList[-1]
    return benchmarkList


def matchBetween(matrixFormSet, filename):
    for matrixForm in matrixFormSet:
        if matrixForm[0] in filename and matrixForm[1] in filename:
            return True
    return False

#Search through library logic files in directory to find a matching kernel
def findMatchingKernel(benchDescriptions, architecture, directoryPath):
    #Whether a benchmark description has found a match
    print("Searching %s for logic files..." % directoryPath)
    foundMatch = [False for x in range(len(benchDescriptions))]

    matrixFormSet = set()
    for bench in benchDescriptions:
        matrixFormSet.add((bench.matrixA, bench.matrixB))

    fileList = []
    for filename in os.listdir(directoryPath):
        if filename.endswith(".yaml") \
        and architecture in filename \
        and matchBetween(matrixFormSet, filename): 
            fileList.append(filename)
    
    success = False
    #Find a kernel with a matching problem size in the file list
    for filename in fileList:
        foundMatchingKernel = False
        f = open(directoryPath + filename, 'r')
        if f.closed:
            sys.exit("File %s could not be opened" % directoryPath + filename)
        libraryLogic = yaml.safe_load(f)
        f.close()
        logicList = libraryLogic[exactLogicListIdx]
        for i in range(len(benchDescriptions)):
            bench = benchDescriptions[i]
            if bench.matrixA in filename and bench.matrixB in filename:
                #Iterate through kernels to find one with matching size
                for [problemInfo, winningKernelInfo] in logicList:
                    foundMatchingKernel = \
                        problemInfo[mIdx]     == bench.m and \
                        problemInfo[nIdx]     == bench.n and \
                        problemInfo[kIdx]     == bench.k and \
                        problemInfo[batchIdx] == bench.batch_count
                    if foundMatchingKernel: break

                if foundMatchingKernel:
                    #print("Match found for %s with speed of %s GFLOPS" % \
                    #    (                  bench,           winningKernelInfo[1]))
                    print("Match found for %s in file %s" % \
                        (                  bench,      filename))
                    foundMatch[i] = True
    return foundMatch

def findBenchmarkInFile(problemDescriptions):
    pass


def main(argv):

    #print(parseOptions("-h -c 1 -a 2 --help --doobeedoo 7".split(), "ha:b:c:", ["help", "doobeedoo="]))
    #return 

    try:
        optdict = parseOptions(argv, "f:a:cui", ["help"])
        if 'help' in optdict.keys():
            print(helpMessage)
            sys.exit()
    except KeyError:
        sys.exit(usageMessage)
    except ParseOptionError:
        sys.exit(usageMessage)

    logpath = ""
    if 'f' in optdict.keys():
        logpath = optdict['f']
    else:
        sys.exit(usageMessage)
    architecture = ""
    if 'a' in optdict.keys():
        architecture = optdict['a']

    cloneInternal = True if 'i' in optdict.keys() else False

    directoryPath="./rocBLASTemp"     #Default path
    libraryPathExtension="/library/src/blas3/Tensile/Logic/asm_full/"
    if not os.path.isdir(directoryPath):
        if not os.path.isfile(directoryPath):
            print("Created temporary directory %s" % directoryPath)
            shellCmd("mkdir %s" % directoryPath)
            cloneRepository(directoryPath, cloneInternal)
        else:
            sys.exit("Directory %s is a file" % directoryPath)

    #Benchmarked problem sizes in log file
    benchDescriptions = loadBenchmarkDescriptions(logpath)

    checkoutInstalledBranch(directoryPath)
    localMatches = findMatchingKernel( \
        benchDescriptions,             \
        architecture,                  \
        directoryPath+libraryPathExtension)

    if 'u' in optdict.keys():
        checkoutMostRecentBranch(directoryPath)
        recentMatches = findMatchingKernel( \
            benchDescriptions,              \
            architecture,                   \
            directoryPath+libraryPathExtension)

        print("Matches in latest version of rocBLAS but not current:")
        for i in range(len(benchDescriptions)):
            if recentMatches[i] and not localMatches[i]:
                b = benchDescriptions[i]
                print("\tm = %d n = %d k = %d batch_count %d matrixA = %s matrixB = %s" \
                    %(        b.m,   b.n,   b.k,           b.batch_count, b.matrixA, b.matrixB))


    #Remove temporary directory only if it wasn't provided by user
    if 'c' in optdict.keys():
        print("Removing temporary directory")
        shellCmd("rm -rf %s" % directoryPath)

if __name__=="__main__":
    main(sys.argv[1:])
