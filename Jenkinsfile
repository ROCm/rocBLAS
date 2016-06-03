node('rocm') {
  //     sh 'env | sort'
    def scm_dir = pwd()
    def build_dir = "${scm_dir}/../build"
    dir("${scm_dir}") {
      stage 'Clone'
      checkout scm
    }
    dir("${build_dir}") {
      stage 'Configure'
        sh "cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_LIBRARY=ON ${scm_dir}"
      stage 'Build'
        sh 'make -j 8'
      stage 'Package'
      // sh 'make package'
        archive includes: '*.deb'
    }
}
