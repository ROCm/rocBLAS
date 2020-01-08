import sys
sys.path.append('/opt/rocm/bin')
import rocm_smi as smi
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d')
    args = parser.parse_args()
    
    if args.d is not None:
        deviceNum = int(args.d)
        deviceList = smi.listDevices(True)
        print("Device List:", deviceList)
        
        device = deviceList[deviceNum]
        print("Device", deviceNum, ":", device)
        
        print("showAllConciseHw")
        smi.showAllConciseHw(deviceList[1:])
        
        print("showProductName")
        smi.showProductName(deviceList)
        
        print("showClocks")
        smi.showClocks(deviceList)
        
        print("getCurrentClock sclk")
        sclk = smi.getCurrentClock(device, 'sclk', 'freq')
        print(sclk)
        
        print("getCurrentClock mclk")
        mclk = smi.getCurrentClock(device, 'mclk', 'freq')
        print(mclk)
