#!/usr/bin/env bash

#use readlink rather than realpath for CentOS 6.10 support
ROCBLAS_SRC_PATH=`dirname "$(readlink -m $0)"`

/bin/ln -fs ../../.githooks/pre-commit "$(dirname "$0")/.git/hooks/"

# #################################################
# helper functions
# #################################################
function display_help()
{
cat <<EOF
rocBLAS build & installation helper script
  $0 <options>
      -h | --help                Print this help message
      -i | --install             Install after build
      -d | --dependencies        Install build dependencies
           --cleanup             Removes intermediary build artifacts after successful build to reduce disk usage
      -c | --clients             Build library clients too (combines with -i & -d)
           --clients-only        Build only clients with a pre-built library
           --library-path        When only building clients, the path to the pre-built rocBLAS library (default is /opt/rocm/rocblas)
      -g | --debug               Set -DCMAKE_BUILD_TYPE=Debug (default is =Release)
      -f | --fork                GitHub fork to use, e.g., ROCmSoftwarePlatform or MyUserName
      -b | --branch              GitHub branch or tag to use, e.g., develop, mybranch or <commit hash>
      -l | --logic               Set Tensile logic target, e.g., asm_full, asm_lite, etc.
      -a | --architecture        Set GPU architecture target(s), e.g., all, gfx000, gfx900, gfx906:xnack-;gfx908:xnack-
      -o | --cov                 Set Tensile code_object_version (V2 or V3)
      -t | --test_local_path     Use a local path for Tensile instead of remote GIT repo
           --cpu_ref_lib         Specify library to use for CPU reference code in testing (blis or lapack)
           --[no-]hip-clang      Whether to build library for amdgpu backend using hip-clang
           --[no-]merge-files    Whether to enable Tensile_MERGE_FILES (default is enable)
           --build_dir           Specify path of output directory relative to the current directory (default is ./build). Also supports absolute path to output directory.
      -n | --no-tensile          Build subset of library that does not require Tensile
      -s | --tensile-host        Build with Tensile host
      -r | --no-tensile-host     Do not build with Tensile host
      -u | --use-custom-version  Ignore Tensile version and just use the Tensile tag
           --use-cuda            Uses installed cuda version instead of rocm stack
           --skipldconf          Skip ld.so.conf entry
           --static              Create static library instead of shared library
      -v | --rocm-dev            Set specific rocm-dev version
           --[no-]msgpack        Set Tensile backend to use MessagePack
           --cmake_install       Auto Update CMake to minimum version if required
           --codecoverage        Build with code coverage profiling enabled
      -k | --relwithdebinfo      Set -DCMAKE_BUILD_TYPE=RelWithDebInfo
           --address-sanitizer   Build with address sanitizer enabled
EOF
#           --prefix              Specify an alternate CMAKE_INSTALL_PREFIX for cmake
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
    ubuntu|centos|rhel|fedora|sles|opensuse-leap|openEuler)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL, SLES, OpenSUSE-Leap, openEuler and Fedora\n"
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
      elevate_if_not_root apt-get install -y --no-install-recommends ${package}
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

install_msgpack_from_source( )
{
    if [[ ! -d "${build_dir}/deps/msgpack-c" ]]; then
      pushd .
      mkdir -p ${build_dir}/deps
      cd ${build_dir}/deps
      git clone -b cpp-3.0.1 https://github.com/msgpack/msgpack-c.git
      cd msgpack-c
      CXX=${cxx} CC=${cc} ${cmake_executable} -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF .
      make
      elevate_if_not_root make install
      popd
    fi
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
# prereq: ${tensile_msgpack_backend} must be defined before calling
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

  # dependencies needed to build the rocblas library
  local library_dependencies_ubuntu=( "make" "libssl-dev"
                                      "python3" "python3-yaml" "python3-venv" "python3*-pip" )
  local library_dependencies_centos_rhel=( "epel-release" "openssl-devel"
                                      "make" "rpm-build"
                                      "python34" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_centos_rhel_8=( "epel-release" "openssl-devel"
                                      "make" "rpm-build"
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )
  local library_dependencies_fedora=( "make" "rpm-build"
                                      "python34" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" "libcxx-devel" )
  local library_dependencies_sles=(   "make" "libopenssl-devel" "python3-PyYAML" "python3-virtualenv"
                                      "gcc-c++" "libcxxtools9" "rpm-build" )
  local library_dependencies_openEuler=( "openssl-devel"
                                      "make" "rpm-build"
                                      "python3" "python3*-PyYAML" "python3-virtualenv"
                                      "gcc-c++" )

  if [[ "${tensile_msgpack_backend}" == true ]]; then
    library_dependencies_ubuntu+=("libmsgpack-dev")
    library_dependencies_fedora+=("msgpack-devel")
  fi

  # wget is needed for msgpack in this case
  if [[ ("${ID}" == "ubuntu") && ("${VERSION_ID}" == "16.04") && "${tensile_msgpack_backend}" == true ]]; then
    if ! $(dpkg -s "libmsgpackc2" &> /dev/null) || $(dpkg --compare-versions $(dpkg-query -f='${Version}' --show libmsgpackc2) lt 2.1.5-1); then
      library_dependencies_ubuntu+=("wget")
    fi
  fi

  # wget is needed for cmake
  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt 3.16.8); then
    if $update_cmake == true; then
      library_dependencies_ubuntu+=("wget")
      library_dependencies_centos_rhel+=("wget")
      library_dependencies_centos_rhel_8+=("wget")
      library_dependencies_fedora+=("wget")
      library_dependencies_sles+=("wget")
      library_dependencies_openEuler+=("wget")
    fi
  fi

  # dependencies to build the client
  local client_dependencies_ubuntu=( "gfortran" "libomp-dev" "libboost-program-options-dev" )
  local client_dependencies_centos_rhel=( "devtoolset-7-gcc-gfortran" "libgomp" "boost-devel" )
  local client_dependencies_centos_rhel_8=( "gcc-gfortran" "libgomp" "boost-devel" )
  local client_dependencies_fedora=( "gcc-gfortran" "libgomp" "boost-devel" )
  local client_dependencies_sles=( "gcc-fortran" "libgomp1" "libboost_program_options1_66_0-devel" )
  local client_dependencies_openEuler=( "gcc-gfortran" "libgomp" "boost-devel" )

  # wget is needed for blis
  if [[ "${cpu_ref_lib}" == blis ]] && [[ ! -e "${build_dir}/deps/blis/lib/libblis.so" ]]; then
    client_dependencies_ubuntu+=("wget")
    client_dependencies_centos_rhel+=("wget")
    client_dependencies_centos_rhel_8+=("wget")
    client_dependencies_fedora+=("wget")
    client_dependencies_sles+=("wget")
    client_dependencies_openEuler+=("wget")
  fi

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt-get update
      install_apt_packages "${library_dependencies_ubuntu[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_apt_packages "${client_dependencies_ubuntu[@]}"
      fi
      ;;

    centos|rhel)
      if [[ ( "${VERSION_ID}" -ge 8 ) ]]; then
        install_yum_packages "${library_dependencies_centos_rhel_8[@]}"

        if [[ "${build_clients}" == true ]]; then
          install_yum_packages "${client_dependencies_centos_rhel_8[@]}"
        fi
      else
  #     yum -y update brings *all* installed packages up to date
  #     without seeking user approval
  #     elevate_if_not_root yum -y update
        install_yum_packages "${library_dependencies_centos_rhel[@]}"

        if [[ "${build_clients}" == true ]]; then
          install_yum_packages "${client_dependencies_centos_rhel[@]}"
        fi
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_dnf_packages "${client_dependencies_fedora[@]}"
      fi
      ;;

    sles|opensuse-leap)
       install_zypper_packages "${library_dependencies_sles[@]}"

        if [[ "${build_clients}" == true ]]; then
            install_zypper_packages "${client_dependencies_sles[@]}"
        fi
        ;;

    openEuler)
       install_yum_packages "${library_dependencies_openEuler[@]}"

        if [[ "${build_clients}" == true ]]; then
            install_yum_packages "${client_dependencies_openEuler[@]}"
        fi
        ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL, SLES, OpenSUSE-Leap, and Fedora"
      exit 2
      ;;
  esac

  if [[ ("${ID}" == "ubuntu") && ("${VERSION_ID}" == "16.04") && "${tensile_msgpack_backend}" == true ]]; then
    # On Ubuntu 16.04, the version of msgpack provided in the repository is outdated, so a newer version
    # must be manually downloaded and installed.  Trying to match or exceed Ubuntu 18 default
    if ! $(dpkg -s "libmsgpackc2" &> /dev/null) || $(dpkg --compare-versions $(dpkg-query -f='${Version}' --show libmsgpackc2) lt 2.1.5-1); then
      wget -nv -P ./ "http://ftp.us.debian.org/debian/pool/main/m/msgpack-c/libmsgpackc2_3.0.1-3_amd64.deb"
      wget -nv -P ./ "http://ftp.us.debian.org/debian/pool/main/m/msgpack-c/libmsgpack-dev_3.0.1-3_amd64.deb"
      elevate_if_not_root dpkg -i ./libmsgpackc2_3.0.1-3_amd64.deb ./libmsgpack-dev_3.0.1-3_amd64.deb
      rm libmsgpack-dev_3.0.1-3_amd64.deb libmsgpackc2_3.0.1-3_amd64.deb
    fi
  fi
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

# /etc/*-release files describe the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
elif [[ -e "/etc/centos-release" ]]; then
  ID=$(cat /etc/centos-release | awk '{print tolower($1)}')
  VERSION_ID=$(cat /etc/centos-release | grep -oP '(?<=release )[^ ]*' | cut -d "." -f1)
else
  echo "This script depends on the /etc/*-release files"
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
gpu_architecture=all
tensile_cov=
tensile_fork=
tensile_merge_files=
tensile_tag=
tensile_test_local_path=
tensile_version=
build_library=true
build_cleanup=false
build_clients=false
use_cuda=false
build_tensile=true
build_tensile_host=true
cpu_ref_lib=blis
build_release=true
build_hip_clang=true
#use readlink rather than realpath for CentOS 6.10 support
build_dir=$(readlink -m ./build)
skip_ld_conf_entry=false
static_lib=false
tensile_msgpack_backend=true
update_cmake=false
build_codecoverage=false
build_release_debug=false
build_address_sanitizer=false

rocm_path=/opt/rocm
if ! [ -z ${ROCM_PATH+x} ]; then
    rocm_path=${ROCM_PATH}
fi

library_dir_installed=${rocm_path}/rocblas

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,cleanup,clients,clients-only,dependencies,debug,hip-clang,no-hip-clang,merge-files,no-merge-files,no_tensile,no-tensile,tensile-host,no-tensile-host,msgpack,no-msgpack,library-path:,logic:,architecture:,cov:,fork:,branch:,build_dir:,test_local_path:,cpu_ref_lib:,use-custom-version:,skipldconf,static,use-cuda,rocm-dev:,cmake_install,codecoverage,relwithdebinfo,address-sanitizer --options nsrhicdgkl:a:o:f:b:t:u:v: -- "$@")
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
    --cleanup)
        build_cleanup=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    --clients-only)
        build_library=false
        build_clients=true
        shift ;;
    --library-path)
        library_dir_installed=${2}
        shift 2 ;;
    -g|--debug)
        build_release=false
        shift ;;
    -l|--logic)
        tensile_logic=${2}
        shift 2 ;;
    -a|--architecture)
        gpu_architecture=${2}
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
    -n|--no_tensile|--no-tensile)
        build_tensile=false
        shift ;;
    -s|--tensile-host)
        build_tensile_host=true
        shift ;;
    -r|--no-tensile-host)
        build_tensile_host=false
        shift ;;
    --build_dir)
#use readlink rather than realpath for CentOS 6.10 support
        build_dir=$(readlink -m ${2})
        shift 2;;
    --use-cuda)
        use_cuda=true
        shift ;;
    --cpu_ref_lib)
        cpu_ref_lib=${2}
        shift 2 ;;
    --hip-clang)
        build_hip_clang=true
        shift ;;
    --no-hip-clang)
        build_hip_clang=false
        shift ;;
    --merge-files)
        tensile_merge_files=true
        shift ;;
    --no-merge-files)
        tensile_merge_files=false
        shift ;;
    --skipldconf)
        skip_ld_conf_entry=true
        shift ;;
    --static)
        static_lib=true
        shift ;;
    -u|--use-custom-version)
        tensile_version=${2}
        shift 2;;
    -v|--rocm-dev)
        custom_rocm_dev=${2}
        shift 2;;
    --prefix)
        install_prefix=${2}
        shift 2 ;;
    --msgpack)
        tensile_msgpack_backend=true
        shift ;;
    --no-msgpack)
        tensile_msgpack_backend=false
        shift ;;
    --cmake_install)
        update_cmake=true
        shift ;;
    --codecoverage)
        build_codecoverage=true
        shift ;;
    -k|--relwithdebinfo)
        build_release=false
        build_release_debug=true
        shift ;;
    --address-sanitizer)
        build_address_sanitizer=true
        shift ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

if [[ -z $tensile_cov ]]; then
    if [[ $build_hip_clang == true ]]; then
        tensile_cov=V3
    else
        tensile_cov=V2
    fi
fi

set -x

if [[ "${cpu_ref_lib}" == blis ]]; then
  LINK_BLIS=true
elif [[ "${cpu_ref_lib}" == lapack ]]; then
  LINK_BLIS=false
else
  echo "Currently the only CPU library options are blis and lapack"
      exit 2
fi

printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

install_blis()
{
    #Download prebuilt AMD multithreaded blis
    if [[ "${cpu_ref_lib}" == blis ]] && [[ ! -e "./blis/lib/libblis.so" ]]; then
      case "${ID}" in
          centos|rhel|sles|opensuse-leap|openEuler)
              wget -nv -O blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-centos-2.0.tar.gz
              ;;
          ubuntu)
              wget -nv -O blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz
              ;;
          *)
              echo "Unsupported OS for this script"
              wget -nv -O blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz
              ;;
      esac

      tar -xvf blis.tar.gz
      rm -rf blis/amd-blis-mt
      mv amd-blis-mt blis
      rm blis.tar.gz
      cd blis/lib
      ln -sf libblis-mt.so libblis.so
    fi
}

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
  rm -rf ${build_dir}/release
elif [[ "${build_release_debug}" == true ]]; then
  rm -rf ${build_dir}/release-debug
else
  rm -rf ${build_dir}/debug
fi

# Default cmake executable is called cmake
cmake_executable=cmake

if [[ "${build_hip_clang}" == true ]]; then
  cxx="hipcc"
  cc="hipcc"
  fc="gfortran"
else
  echo "Currently the only suppported compiler is hip-clang."
  exit 2
fi

# If user provides custom ${rocm_path} path for hcc it has lesser priority,
# but with hip-clang existing path has lesser priority to avoid use of installed clang++ by tensile
if [[ "${build_hip_clang}" == true ]]; then
  export PATH=${rocm_path}/bin:${rocm_path}/hip/bin:${rocm_path}/llvm/bin:${PATH}
fi

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then
  CMAKE_VERSION=$(cmake --version | grep -oP '(?<=version )[^ ]*' )

  install_packages

  if [ -z "$CMAKE_VERSION"] || $(dpkg --compare-versions $CMAKE_VERSION lt 3.16.8); then
      if $update_cmake == true; then
        CMAKE_REPO="https://github.com/Kitware/CMake/releases/download/v3.16.8/"
        wget -nv ${CMAKE_REPO}/cmake-3.16.8.tar.gz
        tar -xvf cmake-3.16.8.tar.gz
        cd cmake-3.16.8
        ./bootstrap --prefix=/usr --no-system-curl --parallel=16
        make -j16
        sudo make install
        cd ..
        rm -rf cmake-3.16.8.tar.gz cmake-3.16.8
      else
          echo "rocBLAS requires CMake version >= 3.16.8 and CMake version ${CMAKE_VERSION} is installed. Run install.sh again with --cmake_install flag and CMake version ${CMAKE_VERSION} will be uninstalled and CMake version 3.16.8 will be installed"
          exit 2
      fi
  fi

  # cmake is needed to install msgpack
  case "${ID}" in
    centos|rhel|sles|opensuse-leap|openEuler)
      if [[ "${tensile_msgpack_backend}" == true ]]; then
        install_msgpack_from_source
      fi
      ;;
  esac

  if [[ "${build_clients}" == true ]]; then

    # The following builds googletest & lapack from source, installs into cmake default /usr/local
    pushd .
    printf "\033[32mBuilding \033[33mgoogletest & lapack\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    CXX=${cxx} CC=${cc} FC=${fc} ${cmake_executable} -DBUILD_BOOST=OFF ${ROCBLAS_SRC_PATH}/deps
    make -j$(nproc)
    elevate_if_not_root make install_deps
    install_blis
    popd
  fi
elif [[ "${build_clients}" == true ]]; then
  pushd .
  mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
  install_blis
  popd
fi

pushd .
  # #################################################
  # configure & build
  # #################################################
  cmake_common_options=""
  cmake_client_options=""

  cmake_common_options="${cmake_common_options} -DROCM_PATH=${rocm_path} -DAMDGPU_TARGETS=${gpu_architecture}"

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  elif [[ "${build_release_debug}" == true ]]; then
    mkdir -p ${build_dir}/release-debug/clients && cd ${build_dir}/release-debug
    cmake_common_options="${cmake_common_options}  -DCMAKE_BUILD_TYPE=RelWithDebInfo"
  else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  # code coverage
  if [[ "${build_codecoverage}" == true ]]; then
      if [[ "${build_release}" == true ]]; then
          echo "Code coverage is disabled in Release mode, to enable code coverage select either Debug mode (-g|--debug) or RelWithDebInfo mode (-k|--relwithdebinfo); aborting";
          exit 1
      fi
      cmake_common_options="${cmake_common_options} -DBUILD_CODE_COVERAGE=ON"
  fi

  if [[ "${static_lib}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_SHARED_LIBS=OFF"
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

  if [[ "${skip_ld_conf_entry}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DROCM_DISABLE_LDCONFIG=ON"
  fi

  if [[ -n "${tensile_version}" ]]; then
    cmake_common_options="${cmake_common_options} -DTENSILE_VERSION=${tensile_version}"
  fi

  tensile_opt=""
  if [[ "${build_tensile}" == false ]]; then
    tensile_opt="${tensile_opt} -DBUILD_WITH_TENSILE=OFF"
   else
    tensile_opt="${tensile_opt} -DTensile_LOGIC=${tensile_logic} -DTensile_CODE_OBJECT_VERSION=${tensile_cov}"
  fi

  if [[ "${build_tensile_host}" == false ]]; then
    tensile_opt="${tensile_opt} -DBUILD_WITH_TENSILE_HOST=OFF"
  fi

  if [[ "${build_tensile_host}" == true ]]; then
    tensile_opt="${tensile_opt} -DBUILD_WITH_TENSILE_HOST=ON"
  fi
  if [[ "${tensile_merge_files}" == false ]]; then
    tensile_opt="${tensile_opt} -DTensile_MERGE_FILES=OFF"
  fi

  if [[ "${tensile_msgpack_backend}" == true ]]; then
    tensile_opt="${tensile_opt} -DTensile_LIBRARY_FORMAT=msgpack"
  else
    tensile_opt="${tensile_opt} -DTensile_LIBRARY_FORMAT=yaml"
  fi

  cmake_common_options="${cmake_common_options} ${tensile_opt}"


  if [[ "${build_clients}" == true ]]; then
    cmake_client_options="${cmake_client_options} -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DLINK_BLIS=${LINK_BLIS} -DBUILD_DIR=${build_dir}"
  fi

  if [[ "${build_library}" == false ]]; then
    cmake_client_options="${cmake_client_options} -DSKIP_LIBRARY=ON -DROCBLAS_LIBRARY_DIR=${library_dir_installed}"
  fi

  if [[ "${build_hip_clang}" == true ]]; then
      cmake_common_options="${cmake_common_options} -DRUN_HEADER_TESTING=OFF"
  fi

  if [[ "${use_cuda}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DUSE_CUDA=ON"
  fi

  if [[ "${build_address_sanitizer}" == true ]]; then
    cmake_common_options="$cmake_common_options -DBUILD_ADDRESS_SANITIZER=ON"
  fi

  # Uncomment for cmake debugging
  # CXX=${compiler} ${cmake_executable} -Wdev --debug-output --trace ${cmake_common_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=rocblas-install -DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path} ../..

  # Build library with AMD toolchain because of existense of device kernels
  if [[ "${build_clients}" == true ]]; then
    CXX=${cxx} CC=${cc} FC=${fc} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=rocblas-install -DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path} ${ROCBLAS_SRC_PATH}
  else
    CXX=${cxx} CC=${cc} ${cmake_executable} ${cmake_common_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=rocblas-install -DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path} ${ROCBLAS_SRC_PATH}
  fi
  check_exit_code "$?"

  if [[ "${build_library}" == true ]]; then
    make -j$(nproc) install
  else
    make -j$(nproc)
  fi
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
        elevate_if_not_root dpkg -i rocblas[_\-]*.deb
      ;;
      centos|rhel)
        elevate_if_not_root yum -y localinstall rocblas-*.rpm
      ;;
      fedora)
        elevate_if_not_root dnf install rocblas-*.rpm
      ;;
      sles|opensuse-leap)
        elevate_if_not_root zypper -n --no-gpg-checks install rocblas-*.rpm
      ;;
    esac

  fi
  check_exit_code "$?"

  if [[ "${build_cleanup}" == true ]]; then
      find -name '*.o' -delete
      find -type d -name '*build_tmp*' -exec rm -rf {} +
      find -type d -name '*_CPack_Packages*' -exec rm -rf {} +
  fi
  check_exit_code "$?"

popd
