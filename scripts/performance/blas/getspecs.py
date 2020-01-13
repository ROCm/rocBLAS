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
    cmd = ["apt", "show", "rocm-libs"]
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


# Get the vbios version for the specified device
def getvbios(devicenum):
    import subprocess, tempfile
    cmd = ["/opt/rocm/bin/rocm-smi", "-v", "-d", str(devicenum)]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            tmp = line[len(searchstr):].strip()[1:]
            pos = tmp.find(":")
            return tmp[pos+1:].strip()

# Get the name of the device from lshw which has index devicenum
def getdeviceinfo(devicenum):
    import subprocess, tempfile, re
    cmd = ["lshw", "-C", "video"]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
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
    # We also use rocm-smi to get more info
    cmd = ["/opt/rocm/bin/rocm-smi", "-i", "-d", str(devicenum)]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            line = line[len(searchstr):].strip()
            line = re.sub(":", "", line)
            line = re.sub("GPU ID", "", line)
            name += " " + line.strip()

    return name

# Get the vram for the specified device
def getvram(devicenum):
    import subprocess, tempfile, re
    cmd = ["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram", "-d", str(devicenum)]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
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
    import subprocess, tempfile, re
    cmd = ["/opt/rocm/bin/rocm-smi", "-p", "-d", str(devicenum)]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            line = line[len(searchstr):].strip()
            skipstr = "Performance Level "
            line = re.sub(":", "", line)[len(skipstr):].strip()
            return line

# Get the memory clock for the specified device
def getmclk(devicenum):
    import subprocess, tempfile, re
    cmd = ["/opt/rocm/bin/rocm-smi", "--showclocks", "-d", str(devicenum)]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    searchstr = "mclk"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if m != None:
            p0 = line.find("(")
            p1 = line.find(")")
            return line[p0+1:p1]

# Get the system clock for the specified device
def getsclk(devicenum):
    import subprocess, tempfile, re
    cmd = ["/opt/rocm/bin/rocm-smi", "--showclocks", "-d", str(devicenum)]
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd,stdout=fout, stderr=ferr)
    p.wait()
    fout.seek(0)
    cout = fout.read()
    searchstr = "sclk"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if m != None:
            p0 = line.find("(")
            p1 = line.find(")")
            return line[p0+1:p1]
