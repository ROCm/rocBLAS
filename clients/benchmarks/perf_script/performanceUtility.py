# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
#
# ########################################################################

#This file contains a number of utilities function which could be independent of
#any specific domain concept

import signal
from subprocess import check_output
import errorHandler
from datetime import datetime

def currentUser():
    try:
        return check_output("who", shell = True).split()[0];
    except:
        print 'Unhandled Exception at performanceUtility::currentUser()'
        raise

#Details: Generate sorted numbers in radices of 2,3 and 5 upto a given upper limit number
def generate235Radices(maxSize):
    sizeList = list()
    i = 0
    j = 0
    k = 0
    SUM = int()
    sumj = int()
    sumk = int()
    sumi = 1
    while(True):
        sumj = 1
        j = 0
        while(True):
            sumk = 1
            k = 0
            while(True):
                SUM = sumi*sumj*sumk
                if ( SUM > maxSize ): break
                sizeList.append(SUM)
                k += 1
                sumk *= 2
            if (k == 0): break
            j += 1
            sumj *= 3
        if ( j == 0 and k == 0): break
        i += 1
        sumi *= 5
    sizeList.sort()
    return sizeList


def timeout(timeout_time, default):
    def timeout_function(f):
        def f2(args):
            def timeout_handler(signum, frame):
                raise errorHandler.TimeoutException()

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_time) # triger alarm in timeout_time seconds
            retval = ""
            try:
                retval = f(args)
            except errorHandler.TimeoutException:
                raise errorHandler.ApplicationException(__file__, errorHandler.TIME_OUT)
            except:
                signal.alarm(0)
                raise
            finally:
                #print 'executing finally'
                signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)
            return retval
        return f2
    return timeout_function


def logTxtOutput(fileName, mode, txt):
    todayFile =  fileName+'-'+datetime.now().strftime('%Y-%b-%d')+'.txt'
    with open(todayFile, mode) as f:
        f.write('------\n'+txt+'\n')

def log(filename, txt):
    with open(filename, 'a') as f:
        f.write(datetime.now().ctime()+'# '+txt+'\n')

