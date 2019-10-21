#!/usr/bin/env bash
# Author: Kent Knox

set -x #echo on

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "rocBLAS build & installation helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
#  echo "    [--prefix] Specify an alternate CMAKE_INSTALL_PREFIX for cmake"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install build dependencies"
  echo "    [-c|--clients] build library clients too (combines with -i & -d)"
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
  echo "    [-f|--fork] GitHub fork to use, ie ROCmSoftwarePlatform or MyUserName"
  echo "    [-b|--branch] GitHub branch or tag to use, ie develop or mybranch or SHA"
  echo "    [-l|--logic] Set tensile logic target (asm_full, asm_lite, etc)"
  echo "    [-o|--cov] Set tensile code_object_version (V2 or V3)"
  echo "    [-t|--test_local_path] Use a local path for tensile instead of remote GIT repot"
#  echo "    [--cuda] build library for cuda backend"
  echo "    [--cpu_ref_lib] specify libary to use for cpu reference code in testing (blis or lapack)"
  echo "    [--hip-clang] build library for amdgpu backend using hip-clang"
  echo "    [-n|--no_tensile] build subset of library that doesn't require tensile (testing)"
  echo "    [-s|--tensile-host] build with tensile host"
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    ubuntu|centos|rhel|fedora|sles)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL and Fedora\n"
        exit 2
        ;;
  esac
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
check_exit_code( )
{
  if (( $1 != 0 )); then
    exit $1
  fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code "$?"
  else
    $@
    check_exit_code "$?"
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $package == *-PyYAML ]] || [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root yum -y --nogpgcheck install ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $package == *-PyYAML ]] || [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root dnf install -y ${package}
    fi
  done
}

install_zypper_packages( )
{
    package_dependencies=("$@")
    for package in "${package_dependencies[@]}"; do
        if [[ $(rpm -q ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
            printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
            elevate_if_not_root zypper install -y ${package}
        fi
    done
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
install_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  if [ -z ${build_clients+foo} ]; then
    printf "install_packages(): \$build_clients must be set\n"
    exit 2
  fi

  # dependencies needed for rocblas and clients to build
  local library_dependencies_ubuntu=( "make" "cmake-curses-gui" "pkg-config"
                                      "python2.7" "python3" "python-yaml" "python3-yaml"
                                      "llvm-6.0-dev" "libomp-dev"
                                      "hip_hcc" "rocm_smi64" "zlib1g-dev")
  local library_dependencies_centos=( "epel-release"
                                      "make" "cmake3" "rpm-build"
                                      "python34" "PyYAML" "python3*-PyYAML"
                                      "gcc-c++" "llvm7.0-devel" "llvm7.0-static"
                                      "hip_hcc" "rocm_smi64" "libgomp" "zlib-devel" )
  local library_dependencies_fedora=( "make" "cmake" "rpm-build"
                                      "python34" "PyYAML" "python3*-PyYAML"
                                      "gcc-c++" "libcxx-devel" "libgomp"
                                      "hip_hcc" "rocm_smi64" "zlib-devel" )
  local library_dependencies_sles=(   "make" "cmake" "python3-PyYAM"
                                      "hip_hcc" "gcc-c++" "libcxxtools9" "rpm-build" )

  if [[ "${build_cuda}" == true ]]; then
    # Ideally, this could be cuda-cublas-dev, but the package name has a version number in it
    library_dependencies_ubuntu+=( "cuda" )
    library_dependencies_centos+=( "" ) # how to install cuda on centos?
    library_dependencies_fedora+=( "" ) # how to install cuda on fedora?
    library_dependencies_sles+=( "" )
  fi

  local client_dependencies_ubuntu=( "gfortran" "libboost-program-options-dev" "libomp-dev")
  local client_dependencies_centos=( "gcc-gfortran" "boost-devel" "libgomp")
  local client_dependencies_fedora=( "gcc-gfortran" "boost-devel" "libgomp")
  local client_dependencies_sles=( "gcc-fortran" "boost-devel" "libboost_program_options1_66_0-devel" "libgomp1")

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_apt_packages "${client_dependencies_ubuntu[@]}"
      fi
      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      install_yum_packages "${library_dependencies_centos[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_yum_packages "${client_dependencies_centos[@]}"
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_dnf_packages "${client_dependencies_fedora[@]}"
      fi
      ;;

    sles)
       install_zypper_packages "${client_dependencies_sles[@]}"

        if [[ "${build_clients}" == true ]]; then
            install_zypper_packages "${client_dependencies_sles[@]}"
        fi
        ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
install_prefix=rocblas-install
tensile_logic=asm_full
tensile_cov=V2
tensile_fork=
tensile_tag=
tensile_test_local_path=
build_clients=false
build_cuda=false
build_tensile=true
build_tensile_host=false
cpu_ref_lib=blis
build_release=true
build_hip_clang=false

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,dependencies,debug,hip-clang,no_tensile,tensile_host,logic:,cov:,fork:,branch:test_local_path:,cpu_ref_lib: --options nshicdgl:o:f:b:t: -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -i|--install)
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    -l|--logic)
        tensile_logic=${2}
        shift 2 ;;
    -o|--cov)
        tensile_cov=${2}
        shift 2 ;;
    -f|--fork)
        tensile_fork=${2}
        shift 2 ;;
    -b|--branch)
        tensile_tag=${2}
        shift 2 ;;
    -t|--test_local_path)
        tensile_test_local_path=${2}
        shift 2 ;;
    -n|--no_tensile)
        build_tensile=false
        shift ;;
    -s|--tensile-host)
        build_tensile_host=true
        shift ;;
    --cuda)
        build_cuda=true
        shift ;;
    --cpu_ref_lib)
        cpu_ref_lib=${2}
        shift 2 ;;
    --hip-clang)
        build_hip_clang=true
        tensile_cov=V3
        shift ;;
    --prefix)
        install_prefix=${2}
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

if [[ "${cpu_ref_lib}" == blis ]]; then
  LINK_BLIS=true
elif [[ "${cpu_ref_lib}" == lapack ]]; then
  LINK_BLIS=false
else
  echo "Currently the only CPU library options are blis and lapack"
      exit 2
fi

build_dir=./build
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
  rm -rf ${build_dir}/release
else
  rm -rf ${build_dir}/debug
fi

# Default cmake executable is called cmake
cmake_executable=cmake

case "${ID}" in
  centos|rhel)
  cmake_executable=cmake3
  ;;
esac

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then

  install_packages

  # The following builds googletest & lapack from source, installs into cmake default /usr/local
  pushd .
    printf "\033[32mBuilding \033[33mgoogletest & lapack\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    ${cmake_executable} -lpthread -DBUILD_BOOST=OFF ../../deps
    make -j$(nproc)
    elevate_if_not_root make install
  popd

fi

if [[ "${cpu_ref_lib}" == blis ]] && [[ ! -f "${build_dir}/deps/blis/lib/libblis.so" ]]; then
  git submodule update --init
  cd extern/blis
  case "${ID}" in
      centos|rhel|sles)
          ./configure --prefix=../../${build_dir}/deps/blis --enable-threading=openmp auto
          ;;
      ubuntu)
          ./configure --prefix=../../${build_dir}/deps/blis --enable-threading=openmp CC=/opt/rocm/hcc/bin/clang auto
          ;;
      *)
          echo "Unsupported OS for this script"
          ./configure --prefix=../../${build_dir}/deps/blis --enable-threading=openmp auto
          ;;
  esac
  make install
  cd ../..
fi

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
export PATH=${PATH}:/opt/rocm/bin

pushd .
  # #################################################
  # configure & build
  # #################################################
  cmake_common_options=""
  cmake_client_options=""

  cmake_common_options="${cmake_common_options} -lpthread -DTensile_LOGIC=${tensile_logic} -DTensile_CODE_OBJECT_VERSION=${tensile_cov}"
  cmake_client_options="-DLINK_BLIS=${LINK_BLIS}"

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  if [[ -n "${tensile_fork}" ]]; then
    cmake_common_options="${cmake_common_options} -Dtensile_fork=${tensile_fork}"
  fi

  if [[ -n "${tensile_tag}" ]]; then
    cmake_common_options="${cmake_common_options} -Dtensile_tag=${tensile_tag}"
  fi

  if [[ -n "${tensile_test_local_path}" ]]; then
    cmake_common_options="${cmake_common_options} -DTensile_TEST_LOCAL_PATH=${tensile_test_local_path}"
  fi


case "${ID}" in
  centos|rhel)
  cmake_common_options="${cmake_common_options} -DCMAKE_FIND_ROOT_PATH=/usr/lib64/llvm7.0/lib/cmake/"
  ;;
esac

  # clients

  tensile_opt=""
    if [[ "${build_tensile}" == false ]]; then
    tensile_opt="${tensile_opt} -DBUILD_WITH_TENSILE=OFF"
  fi

    if [[ "${build_tensile_host}" == true ]]; then
    tensile_opt="${tensile_opt} -DBUILD_WITH_TENSILE_HOST=ON"
  fi

  if [[ "${build_clients}" == true ]]; then
    cmake_client_options="${cmake_client_options} ${tensile_opt} -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON"
  fi

  compiler="hcc"
  if [[ "${build_cuda}" == true || "${build_hip_clang}" == true ]]; then
    compiler="hipcc"
    cmake_common_options="${cmake_common_options} -DTensile_COMPILER=hipcc"
  fi

  # Uncomment for cmake debugging
  # CXX=${compiler} ${cmake_executable} -Wdev --debug-output --trace ${cmake_common_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=rocblas-install -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm ../..

  # Build library with AMD toolchain because of existense of device kernels
  if [[ "${build_clients}" == true ]]; then
    CXX=${compiler} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=rocblas-install -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm ../..
  else
    CXX=${compiler} ${cmake_executable} ${cmake_common_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=rocblas-install -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm ../..
  fi
  check_exit_code "$?"

  make -j$(nproc) install
  check_exit_code "$?"

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    make package
    check_exit_code "$?"

    case "${ID}" in
      ubuntu)
        elevate_if_not_root dpkg -i rocblas-*.deb
      ;;
      centos|rhel)
        elevate_if_not_root yum -y localinstall rocblas-*.rpm
      ;;
      fedora)
        elevate_if_not_root dnf install rocblas-*.rpm
      ;;
      sles)
        elevate_if_not_root zypper --no-gpg-checks in -y install rocblas-*.rpm
      ;;
    esac

  fi
popd
