# rocBLAS

rocBLAS is the [ROCm](https://rocm.docs.amd.com/en/latest) Basic Linear Algebra Subprograms (BLAS)
library. rocBLAS is implemented in the
[HIP programming language](https://github.com/ROCm-Developer-Tools/HIP) and optimized for AMD
GPUs.

## Requirements

You must have ROCm installed on your system before you can install rocBLAS. For information on
ROCm installation and required platform dependencies, refer to the
[ROCm](https://rocm.docs.amd.com/en/latest).

## Documentation

Documentation for rocBLAS is available at
[https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html).

To build documentation locally, use the following code:

```bash
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Install and build

After you install the ROCm package repositories, you can download and install the `rocblas` package
from the system package manager. For example, on Ubuntu you can use the following code:

```bash
sudo apt-get update
sudo apt-get install rocblas
```

On Fedora, you can use the following code:

```bash
sudo dnf install rocblas
sudo dnf install rocblas-devel
```
