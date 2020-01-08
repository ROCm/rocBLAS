import sys
sys.path.append('/opt/rocm/bin')
import rocm_smi as smi
import argparse
import io
from contextlib import redirect_stdout

def parseRocmsmi(func, arg, matchStr):
    f = io.StringIO()
    with redirect_stdout(f):
        if arg is not None:
            func(arg)
        else:
            func()
    lines = f.getvalue().split('\n')
    output = ''
    for line in lines:
        if matchStr in line:
            output = line[line.find(matchStr)+len(matchStr):]
    return output
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d')
    args = parser.parse_args()
    
    if args.d is not None:
        deviceNum = int(args.d)
        print("Inputted Device Argument:", deviceNum)
        
        #get device name from number
        device = smi.parseDeviceNumber(deviceNum)
        
        print("All devices:", smi.listDevices(True))
        print("AMD devices:", smi.listDevices(False))
        
        if smi.isAmdDevice(device):
            deviceList = [device]
            print("\nDevice chosen:", device)
            
            print("\nshowAllConciseHw")
            smi.showAllConciseHw(deviceList)
            
            print("\nshowProductName")
            smi.showProductName(deviceList)
            
            print("\nshowClocks")
            smi.showClocks(deviceList)
            
            print("\ngetCurrentClock sclk")
            sclk = smi.getCurrentClock(device, 'sclk', 'freq')
            print("sclk:", sclk)
            
            print("\ngetCurrentClock mclk")
            mclk = smi.getCurrentClock(device, 'mclk', 'freq')
            print("mclk:", mclk)
            
            print("\nshowVbiosVersion:")
            smi.showVbiosVersion(deviceList)
            vbios = parseRocmsmi(smi.showVbiosVersion, deviceList, 'VBIOS version: ')
            print("VBIOS:", vbios)
            
            print("\nshowId:", vbios)
            smi.showId(deviceList)
            deviceId = parseRocmsmi(smi.showId, deviceList, 'GPU ID: ')
            print("device ID:", deviceId)
            
            print("\nshowPerformanceLevel")
            smi.showPerformanceLevel(deviceList)
            performanceLevel = parseRocmsmi(smi.showPerformanceLevel, deviceList, 'Performance Level: ')
            print("Performance Level:", performanceLevel)
        else:
            print("Selected device is not an AMD GPU")
