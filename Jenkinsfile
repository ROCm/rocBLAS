node('rocm') {
  //     sh 'env | sort'
    def scm_dir = pwd()
    def build_dir_debug = "${scm_dir}/../build/debug"
    def build_dir_release = "${scm_dir}/../build/release"
    dir("${scm_dir}") {
      stage 'Clone'
      checkout scm
    }
    dir("${build_dir_release}") {
      stage 'configure clang release'
        sh "cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DHIP_ROOT=/opt/rocm/hip -DBOOST_ROOT=/opt/boost/clang -DBUILD_WITH_COBALT=OFF ${scm_dir}"
      stage 'Build'
        sh 'make -j 8'
      stage 'Package Debian'
        sh 'cd rocblas-build; make package'
        archive includes: 'rocblas-build/*.deb'
      stage 'samples'
        sh "cd clients-build/samples-build; make; ./example_sscal"
    }
    dir("${build_dir_debug}") {
      stage 'clang-tidy checks'
        sh "cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DHIP_ROOT=/opt/rocm/hip -DBOOST_ROOT=/opt/boost/clang -DBUILD_WITH_COBALT=OFF ${scm_dir}"
        sh 'make'
    }
}
