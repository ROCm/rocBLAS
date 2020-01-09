import sys
sys.path.append('/opt/rocm/bin')
import rocm_smi as smi
import io
from contextlib import redirect_stdout

#get device name from rocm-smi
def getDeviceName(devicenum):
    return smi.parseDeviceNumber(devicenum)

#parse information from rocmsmi
def parseRocmsmi(func, args, searchStr):
    f = io.StringIO()
    with redirect_stdout(f):
        if args is not None:
            func(*args)
        else:
            func()
    lines = f.getvalue().split('\n')
    output = ''
    for line in lines:
        if searchStr in line:
            output = line[line.find(searchStr)+len(searchStr):]
    return output

# Get the hostname
def gethostname():
    import socket
    hostname = socket.gethostname()
    return hostname

# Get the host cpu information
def getcpu():
    import subprocess, tempfile
    cmd = ["lscpu"]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    cpulist = ""
    fout.seek(0)
    cout = fout.read()
    searchstr = "Model name:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            cpulist += line[len(searchstr):].strip()
    return cpulist

# Get the kernel version
def getkernel():
    import subprocess
    cmd = ["uname", "-r"]
    import tempfile
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    return cout.strip()

# Get the host ram size
def getram():
    import subprocess, tempfile, re
    cmd = ["lshw", "-class",  "memory"]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    searchstr = "size:"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if not m == None:
            return line.strip()[len(searchstr):].strip()

# Get the Linux distro information
def getdistro():
    import subprocess, tempfile
    cmd = ["lsb_release", "-a"]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    searchstr = "Description:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            return line[len(searchstr):].strip()

# Get the version number for rocm
def getrocmversion():
    import subprocess, tempfile
    cmd = ["apt", "show", "rocm-dev"]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    searchstr = "Version:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            return line[len(searchstr):].strip()


# Get the vbios version from rocm-smi
def getvbios(devicenum):
    deviceList = [getDeviceName(devicenum)]
    print("\nshowVbiosVersion:")
    smi.showVbiosVersion(deviceList)
    vbios = parseRocmsmi(smi.showVbiosVersion, [deviceList], 'VBIOS version: ')
    print("VBIOS:", vbios)
    return vbios

# Get the device id from rocm-smi
def getdeviceinfo(devicenum):
    deviceList = [getDeviceName(devicenum)]
    print("\nshowId:")
    smi.showId(deviceList)
    deviceId = parseRocmsmi(smi.showId, [deviceList], 'GPU ID: ')
    print("device ID:", deviceId)
    return deviceId

# Get the vram for the specified device
def getvram(devicenum):
    deviceList = [getDeviceName(devicenum)]
    print("\nshowMemInfo")
    smi.showMemInfo(deviceList, ['vram'])
    vram = parseRocmsmi(smi.showMemInfo, [deviceList, ['vram']], 'vram Total Memory (B): ')
    return vram

# Get the performance level for the specified device
def getperflevel(devicenum):
    deviceList = [getDeviceName(devicenum)]
    print("\nshowPerformanceLevel")
    smi.showPerformanceLevel(deviceList)
    performanceLevel = parseRocmsmi(smi.showPerformanceLevel, [deviceList], 'Performance Level: ')
    print("Performance Level:", performanceLevel)
    return performanceLevel

# Get the memory clock for the specified device
def getmclk(devicenum):
    print("\ngetCurrentClock mclk")
    mclk = smi.getCurrentClock(getDeviceName(devicenum), 'mclk', 'freq')
    print("mclk:", mclk)
    return mclk

# Get the system clock for the specified device
def getsclk(devicenum):
    print("\ngetCurrentClock sclk")
    sclk = smi.getCurrentClock(getDeviceName(devicenum), 'sclk', 'freq')
    print("sclk:", sclk)
    return sclk
