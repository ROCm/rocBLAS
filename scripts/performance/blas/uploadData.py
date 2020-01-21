'''
Converts data file from rocBLAS perf script and uploads to InfluxDB
'''

import datetime
import os
import sys
import argparse
from influxdb import InfluxDBClient

gfxArchs = ['gfx803', 'gfx900', 'gfx906', 'gfx908']

def generateMeasurement(gfxArch, measurementName, testName, testType, testReal, testSize, testVal):
        outputDict = {
                        "measurement": "rocblas_" + measurementName.lower(),
                        "tags": {
                            "testname": testName,
                            "datatype": testType,
                            "complex": testReal,
                            "size": testSize,
                            "gfxArch": gfxArch
                        },
                        "time": timestamp,
                        "fields": {
                            "value": testVal
                        }
                    }
        return outputDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', help='gfx architecture')
    parser.add_argument('-f', help='data folder')
    args = parser.parse_args()

    if args.f is not None:
        dataFolder = args.f
    else:
        print("No data folder specified")
        sys.exit(2)

    if args.a in gfxArchs:
        gfxArch = args.a
        print(gfxArch)
        #Host and Port to be defined by Jenkins?
        # influxHost = 'cgy-slowpoke'
        influxHost = '10.216.151.18'
        influxPort = 8086
        influxDatabase = 'test'

        client = InfluxDBClient(influxHost, influxPort, database=influxDatabase)



        timeInfo = dict()
        timeInfo['name'] = 'Time'
        timeInfo['unit'] = 'us'

        perfInfo = dict()
        perfInfo['name'] = 'Performance'
        perfInfo['unit'] = 'GFlops'

        bandwidthInfo = dict()
        bandwidthInfo['name'] = 'Bandwidth'
        bandwidthInfo['unit'] = 'GB/s'

        info = [timeInfo, perfInfo, bandwidthInfo]

        timestamp = datetime.datetime.utcnow()

        json_body = []
        for file in os.listdir(dataFolder):
            if file.endswith(".dat"):
                #list of dicts. each dict is a 'measurement' in influxDB

                print(file)
                nameElems = file.split("_")
                testName = nameElems[0]
                testType = nameElems[1]
                testReal = nameElems[2]
                with open(os.path.join(dataFolder, file), 'r') as f:
                    lines = f.readlines()


                ##Read data from file
                #skip first line
                for line in lines[1:]:
                    #ignore empty lines
                    if line.strip():
                        splitline = line.split()
                        #note that this could potentially be a float
                        testSize = int(splitline[0])
                        currentIndex = 0
                        #in each line iterate through the multiple types of data
                        for i in range(len(info)):
                            currentIndex += 1
                            #read in number of data values following this number
                            numVals = int(splitline[currentIndex])
                            #read in the data values
                            vals = []
                            for j in range(numVals):
                                currentIndex += 1
                                vals.append(float(splitline[currentIndex]))
                            #check if there are values
                            if len(vals) > 0:
                                #take the average of the values
                                testVal = sum(vals) / len(vals)
                                #generate measurement and append
                                measurement = generateMeasurement(gfxArch, info[i]['name'], testName, testType, testReal, testSize, testVal)
                                json_body.append(measurement)

        client.write_points(json_body)
    else:
        print("Invalid gfx architecture")
        sys.exit(2)

