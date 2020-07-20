def _subprocess_helper(cmd, *args, **kwargs):
    import subprocess
    import tempfile
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    cout = ""
    success = False
    try:
        p = subprocess.Popen(cmd, stdout=fout, stderr=ferr, *args, **kwargs)
        p.wait()
        fout.seek(0)
        cout = fout.read()
        success = True
    except subprocess.CalledProcessError:
        pass
    except FileNotFoundError:
        pass
    return success, cout


# Get the hostname
def gethostname():
    import socket
    hostname = socket.gethostname()
    return hostname

# Get the host cpu information
def getcpu():
    cmd = ["lscpu"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    cpulist = ""
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
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    return cout.strip()

# Get the host ram size
def getram():
    import re
    cmd = ["lshw", "-class",  "memory"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "size:"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if not m == None:
            return line.strip()[len(searchstr):].strip()

# Get the Linux distro information
def getdistro():
    cmd = ["lsb_release", "-a"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "Description:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            return line[len(searchstr):].strip()

# Get the version number for rocm
def getrocmversion():
    cmd = ["apt", "show", "rocm-libs"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "Version:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            return line[len(searchstr):].strip()


# Get the vbios version for the specified device
def getvbios(devicenum):
    cmd = ["/opt/rocm/bin/rocm-smi", "-v", "-d", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            tmp = line[len(searchstr):].strip()[1:]
            pos = tmp.find(":")
            return tmp[pos+1:].strip()
    return ""

def getgpuid(devicenum):
    import re
    name = ""
    # We also use rocm-smi to get more info
    cmd = ["/opt/rocm/bin/rocm-smi", "-i", "-d", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            line = line[len(searchstr):].strip()
            line = re.sub(":", "", line)
            line = re.sub("GPU ID", "", line)
            name += " " + line.strip()
            name = name.replace(" ", "")
    return name

# Get the name of the device from lshw which has index devicenum
def getdeviceinfo(devicenum):
    import re
    cmd = ["lshw", "-C", "video"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "-display"
    indices = []
    name = ""
    for idx, line in enumerate(cout.split("\n")):
        if re.search(searchstr, line) != None:
            indices.append(idx)
    for idx, line in enumerate(cout.split("\n")):
        if idx >= indices[devicenum]:
            searchstr = "product:"
            if re.search(searchstr, line) != None:
                pos = line.find(":")
                name += line[pos+1:].strip()
    name += " " + getgpuid(devicenum)
    return name

# Get the vram for the specified device
def getvram(devicenum):
    import re
    cmd = ["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram", "-d", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            line = line[len(searchstr):].strip()
            prestring = "vram :: total:"
            line = re.sub(":", "", line)
            line = re.sub("vram", "", line)
            line = re.sub("total", "", line)
            pos = line.find("used")
            return line[:pos].strip()

# Get the performance level for the specified device
def getperflevel(devicenum):
    import re
    cmd = ["/opt/rocm/bin/rocm-smi", "-p", "-d", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            line = line[len(searchstr):].strip()
            skipstr = "Current Performance Level "
            line = re.sub(":", "", line)[len(skipstr):].strip()
            return line

# Get the memory clock for the specified device
def getmclk(devicenum):
    import re
    cmd = ["/opt/rocm/bin/rocm-smi", "--showclocks", "-d", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "mclk"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if m != None:
            p0 = line.find("(")
            p1 = line.find(")")
            return line[p0+1:p1]

# Get the system clock for the specified device
def getsclk(devicenum):
    import re
    cmd = ["/opt/rocm/bin/rocm-smi", "--showclocks", "-d", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "sclk"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if m != None:
            p0 = line.find("(")
            p1 = line.find(")")
            return line[p0+1:p1]

def getbandwidth(devicenum):
    gpuid = getgpuid(devicenum)
    if gpuid == "0x66af":
        # radeon7: float: 13.8 TFLOPs, double: 3.46 TFLOPs, 1024 GB/s
        return (13.8, 3.46, 1024)
