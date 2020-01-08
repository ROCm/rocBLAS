import sys
sys.path.append('/opt/rocm/bin')
import rocm_smi as smi


deviceList = smi.listDevices(True)
print("Device List:", deviceList)

device = deviceList[0]
print("Device 0:", device)

print("showAllConciseHw")
smi.showAllConciseHw(deviceList)

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
