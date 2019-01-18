#!/usr/bin/env groovy

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    // parameters([booleanParam( name: 'push_image_to_docker_hub', defaultValue: false, description: 'Push rocblas image to rocm docker-hub' )]),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// -- AUXILLARY HELPER FUNCTIONS
// import hudson.FilePath;
import java.nio.file.Path;

////////////////////////////////////////////////////////////////////////
// Check whether job was started by a timer
@NonCPS
def isJobStartedByTimer() {
    def startedByTimer = false
    try {
        def buildCauses = currentBuild.rawBuild.getCauses()
        for ( buildCause in buildCauses ) {
            if (buildCause != null) {
                def causeDescription = buildCause.getShortDescription()
                echo "shortDescription: ${causeDescription}"
                if (causeDescription.contains("Started by timer")) {
                    startedByTimer = true
                }
            }
        }
    } catch(theError) {
        echo "Error getting build cause"
    }

    return startedByTimer
}

////////////////////////////////////////////////////////////////////////
// Return build number of upstream job
@NonCPS
int get_upstream_build_num( )
{
    def upstream_cause = currentBuild.rawBuild.getCause( hudson.model.Cause$UpstreamCause )
    if( upstream_cause == null)
      return 0

    return upstream_cause.getUpstreamBuild()
}

////////////////////////////////////////////////////////////////////////
// Return project name of upstream job
@NonCPS
String get_upstream_build_project( )
{
    def upstream_cause = currentBuild.rawBuild.getCause( hudson.model.Cause$UpstreamCause )
    if( upstream_cause == null)
      return null

    return upstream_cause.getUpstreamProject()
}

////////////////////////////////////////////////////////////////////////
// Calculate the relative path between two sub-directories from a common root
@NonCPS
String g_relativize( String root_string, String rel_source, String rel_build )
{
  Path root_path = new File( root_string ).toPath( )
  Path path_src = root_path.resolve( rel_source )
  Path path_build = root_path.resolve( rel_build )

  return path_build.relativize( path_src ).toString( )
}

////////////////////////////////////////////////////////////////////////
// Construct the relative path of the build directory
void build_directory_rel( project_paths paths, compiler_data hcc_args )
{
//   if( hcc_args.build_config.equalsIgnoreCase( 'release' ) )
//   {
//     paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name + '/release';
//   }
//   else
//   {
//     paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name + '/debug';
//   }
  paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name;
}

////////////////////////////////////////////////////////////////////////
// Lots of images are created above; no apparent way to delete images:tags with docker global variable
def docker_clean_images( String org, String image_name )
{
  // Check if any images exist first grepping for image names
  int docker_images = sh( script: "docker images | grep \"${org}/${image_name}\"", returnStatus: true )

  // The script returns a 0 for success (images were found )
  if( docker_images == 0 )
  {
    // run bash script to clean images:tags after successful pushing
    sh "docker images | grep \"${org}/${image_name}\" | awk '{print \$1 \":\" \$2}' | xargs docker rmi"
  }
}

////////////////////////////////////////////////////////////////////////
// -- BUILD RELATED FUNCTIONS

////////////////////////////////////////////////////////////////////////
// Checkout source code, source dependencies and update version number numbers
// Returns a relative path to the directory where the source exists in the workspace
void checkout_and_version( project_paths paths )
{
  paths.project_src_prefix = paths.src_prefix + '/' + paths.project_name

  dir( paths.project_src_prefix )
  {
    // checkout rocblas
    checkout([
      $class: 'GitSCM',
      branches: scm.branches,
      doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
      extensions: scm.extensions + [[$class: 'CleanCheckout']],
      userRemoteConfigs: scm.userRemoteConfigs
    ])
  }
}

////////////////////////////////////////////////////////////////////////
// This creates the docker image that we use to build the project in
// The docker images contains all dependencies, including OS platform, to build
def docker_build_image( docker_data docker_args, project_paths paths )
{
  String build_image_name = "build-rocblas-hip-artifactory"
  def build_image = null

  dir( paths.project_src_prefix )
  {
    def user_uid = sh( script: 'id -u', returnStdout: true ).trim()

    // Docker 17.05 introduced the ability to use ARG values in FROM statements
    // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
    // build_image = docker.build( "${paths.project_name}/${build_image_name}:latest", "--pull -f docker/${build_docker_file} --build-arg user_uid=${user_uid} --build-arg base_image=${from_image} ." )

    // JENKINS-44836 workaround by using a bash script instead of docker.build()
    sh "docker build -t ${paths.project_name}/${build_image_name}:latest -f docker/${docker_args.build_docker_file} ${docker_args.docker_build_args} --build-arg user_uid=${user_uid} --build-arg base_image=${docker_args.from_image} ."
    build_image = docker.image( "${paths.project_name}/${build_image_name}:latest" )
  }

  return build_image
}

////////////////////////////////////////////////////////////////////////
// This encapsulates the cmake configure, build and package commands
// Leverages docker containers to encapsulate the build in a fixed environment
def docker_build_inside_image( def build_image, compiler_data compiler_args, docker_data docker_args, project_paths paths )
{
  // Construct a relative path from build directory to src directory; used to invoke cmake
  String rel_path_to_src = g_relativize( pwd( ), paths.project_src_prefix, paths.project_build_prefix )

  String build_type_postfix = null
  if( compiler_args.build_config.equalsIgnoreCase( 'release' ) )
  {
    build_type_postfix = ""
  }
  else
  {
    build_type_postfix = "-d"
  }

  build_image.inside( docker_args.docker_run_args )
  {
    withEnv(["CXX=${compiler_args.compiler_path}", 'CLICOLOR_FORCE=1'])
    {
      // Build library & clients
      sh  """#!/usr/bin/env bash
          set -x
          cd ${paths.project_build_prefix}
          ${paths.build_command}
        """
    }

    if( paths.project_name.equalsIgnoreCase( 'rocblas-ubuntu' ) )
    {
      stage('Clang Format')
      {
        sh '''
            find . -iname \'*.h\' \
                -o -iname \'*.hpp\' \
                -o -iname \'*.cpp\' \
                -o -iname \'*.h.in\' \
                -o -iname \'*.hpp.in\' \
                -o -iname \'*.cpp.in\' \
            | grep -v 'build/' \
            | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-3.8 -style=file {} | diff - {}\'
        '''
      }
    }

    stage( "Test ${compiler_args.compiler_name} ${compiler_args.build_config}" )
    {
      // Cap the maximum amount of testing to be a few hours; assume failure if the time limit is hit
      timeout(time: 4, unit: 'HOURS')
      {
        if(isJobStartedByTimer())
        {
          sh """#!/usr/bin/env bash
                set -x
                cd ${paths.project_build_prefix}/build/release/clients/staging
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib ./rocblas-test${build_type_postfix} --gtest_output=xml --gtest_color=yes --gtest_filter=*nightly*-*known_bug* #--gtest_filter=*nightly*
            """
          junit "${paths.project_build_prefix}/build/release/clients/staging/*.xml"
        }
        else
        {
          sh """#!/usr/bin/env bash
                set -x
                cd ${paths.project_build_prefix}/build/release/clients/staging
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib ./example-sscal${build_type_postfix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib ./rocblas-test${build_type_postfix} --gtest_output=xml --gtest_color=yes  --gtest_filter=*quick*:*pre_checkin*-*known_bug* #--gtest_filter=*checkin*
            """
          junit "${paths.project_build_prefix}/build/release/clients/staging/*.xml"
        }
      }

      String docker_context = "${compiler_args.build_config}/${compiler_args.compiler_name}"
      if( compiler_args.compiler_name.toLowerCase( ).startsWith( 'hcc-' ) )
      {
        sh  """#!/usr/bin/env bash
            set -x
            cd ${paths.project_build_prefix}/build/release
            make package
          """

        if( paths.project_name.equalsIgnoreCase( 'rocblas-ubuntu' ) )
        {
          sh  """#!/usr/bin/env bash
              set -x
              rm -rf ${docker_context} && mkdir -p ${docker_context}
              mv ${paths.project_build_prefix}/build/release/*.deb ${docker_context}
              dpkg -c ${docker_context}/*.deb
          """
          archiveArtifacts artifacts: "${docker_context}/*.deb", fingerprint: true
        }
        else if( paths.project_name.equalsIgnoreCase( 'rocblas-fedora' ) )
        {
          sh  """#!/usr/bin/env bash
              set -x
              rm -rf ${docker_context} && mkdir -p ${docker_context}
              mv ${paths.project_build_prefix}/build/release/*.rpm ${docker_context}
              rpm -qlp ${docker_context}/*.rpm
          """
          archiveArtifacts artifacts: "${docker_context}/*.rpm", fingerprint: true
        }
      }

    }
  }

  return void
}

////////////////////////////////////////////////////////////////////////
// This builds a fresh docker image FROM a clean base image, with no build dependencies included
// Uploads the new docker image to internal artifactory
// String docker_test_install( String hcc_ver, String artifactory_org, String from_image, String rocblas_src_rel, String build_dir_rel )
String docker_test_install( compiler_data compiler_args, docker_data docker_args, project_paths rocblas_paths, String job_name )
{
  def rocblas_install_image = null
  String image_name = "rocblas-hip-${compiler_args.compiler_name}-ubuntu-16.04"
  String docker_context = "${compiler_args.build_config}/${compiler_args.compiler_name}"

  stage( "Install ${compiler_args.compiler_name} ${compiler_args.build_config}" )
  {
    //  We copy the docker files into the bin directory where the .deb lives so that it's a clean build everytime
    sh  """#!/usr/bin/env bash
        set -x
        mkdir -p ${docker_context}
        cp -r ${rocblas_paths.project_src_prefix}/docker/* ${docker_context}
      """

    // Docker 17.05 introduced the ability to use ARG values in FROM statements
    // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
    // rocblas_install_image = docker.build( "${job_name}/${image_name}:${env.BUILD_NUMBER}", "--pull -f ${build_dir_rel}/dockerfile-rocblas-ubuntu-16.04 --build-arg base_image=${from_image} ${build_dir_rel}" )

    // JENKINS-44836 workaround by using a bash script instead of docker.build()
    sh """docker build -t ${job_name}/${image_name} --pull -f ${docker_context}/${docker_args.install_docker_file} \
        --build-arg base_image=${docker_args.from_image} ${docker_context}"""
    rocblas_install_image = docker.image( "${job_name}/${image_name}" )

  // NOTE: Don't push to artifactory yet, just test install package

  // The connection to artifactory can fail sometimes, but this should not be treated as a build fail
  //   try
  //   {
  //     // Don't push pull requests to artifactory, these tend to accumulate over time
  //     if( env.BRANCH_NAME.toLowerCase( ).startsWith( 'pr-' ) )
  //     {
  //       println 'Pull Request (PR-xxx) detected; NOT pushing to artifactory'
  //     }
  //     else
  //     {
  //       docker.withRegistry('http://compute-artifactory:5001', 'artifactory-cred' )
  //       {
  //         rocblas_install_image.push( "${env.BUILD_NUMBER}" )
  //         rocblas_install_image.push( 'latest' )
  //       }
  //     }
  //   }
  //   catch( err )
  //   {
  //     currentBuild.result = 'SUCCESS'
  //   }
  }

  return image_name
}

////////////////////////////////////////////////////////////////////////
// Uploads the new docker image to the public docker-hub
def docker_upload_dockerhub( String local_org, String image_name, String remote_org )
{
  stage( 'docker-hub' )
  {
    // Do not treat failures to push to docker-hub as a build fail
    try
    {
      sh  """#!/usr/bin/env bash
          set -x
          echo inside sh
          docker tag ${local_org}/${image_name} ${remote_org}/${image_name}
        """

      docker_hub_image = docker.image( "${remote_org}/${image_name}" )

      docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-cred' )
      {
        docker_hub_image.push( "${env.BUILD_NUMBER}" )
        docker_hub_image.push( 'latest' )
      }
    }
    catch( err )
    {
      currentBuild.result = 'SUCCESS'
    }
  }
}

////////////////////////////////////////////////////////////////////////
// hip_integration_testing
// This function sets up compilation and testing of HiP on a compiler downloaded from an upstream build
// Integration testing is centered around docker and constructing clean test environments every time

// NOTES: I have implemeneted integration testing 3 different ways, and I've come to the conclusion nothing is perfect
// 1.  I've tried having HCC push the test compiler to artifactory, and having HiP download the test docker image from artifactory
//     a.  The act of uploading and downloading images from artifactory takes minutes
//     b.  There is no good way of deleting images from a repository.  You have to use an arcane CURL command and I don't know how
//        to keep the password secret.  These test integration images are meant to be ephemeral.
// 2.  I tried 'docker save' to export a docker image into a tarball, and transfering the image through 'copy artifacts plugin'
//     a.  The HCC docker image uncompressed is over 1GB
//     b.  Compressing the docker image takes even longer than uploading the image to artifactory
// 3.  Download the HCC .deb and dockerfile through 'copy artifacts plugin'.  Create a new HCC image on the fly
//     a.  There is inefficency in building a new ubuntu image and installing HCC twice (once in HCC build, once here)
//     b.  This solution doesn't scale when we start testing downstream libraries

// I've implemented solution #3 above, probably transitioning to #2 down the line (probably without compression)
String hip_integration_testing( String inside_args, String job, String build_config )
{
  // Attempt to make unique docker image names for each build, to support concurrent builds
  // Mangle docker org name with upstream build info
  String testing_org_name = 'hip-test-' + get_upstream_build_project( ).replaceAll('/','-') + '-' + get_upstream_build_num( )

  // Tag image name with this build number
  String hip_test_image_name = "hip:${env.BUILD_NUMBER}"

  def rocblas_integration_image = null

  dir( 'integration-testing' )
  {
    deleteDir( )

    // This invokes 'copy artifact plugin' to copy archived files from upstream build
    step([$class: 'CopyArtifact', filter: 'archive/**/*.deb, docker/dockerfile-*',
      fingerprintArtifacts: true, projectName: get_upstream_build_project( ), flatten: true,
      selector: [$class: 'TriggeredBuildSelector', allowUpstreamDependencies: false, fallbackToLastSuccessful: false, upstreamFilterStrategy: 'UseGlobalSetting'],
      target: '.' ])

    docker.build( "${testing_org_name}/${hip_test_image_name}", "-f dockerfile-hip-ubuntu-16.04 ." )
  }

  // Checkout source code, dependencies and version files
  String rocblas_src_rel = checkout_and_version( job )

  // Conctruct a binary directory path based on build config
  String rocblas_bin_rel = build_directory_rel( build_config )

  // Build rocblas inside of the build environment
  rocblas_integration_image = docker_build_image( job, testing_org_name, '', rocblas_src_rel, "${testing_org_name}/${hip_test_image_name}" )

  docker_build_inside_image( rocblas_integration_image, inside_args, job, '', build_config, rocblas_src_rel, rocblas_bin_rel )

  docker_clean_images( testing_org_name, '*' )
}

// Docker related variables gathered together to reduce parameter bloat on function calls
class docker_data implements Serializable
{
  String from_image
  String build_docker_file
  String install_docker_file
  String docker_run_args
  String docker_build_args
}

// Docker related variables gathered together to reduce parameter bloat on function calls
class compiler_data implements Serializable
{
  String compiler_name
  String build_config
  String compiler_path
}

// Paths variables bundled together to reduce parameter bloat on function calls
class project_paths implements Serializable
{
  String project_name
  String src_prefix
  String project_src_prefix
  String build_prefix
  String project_build_prefix
  String build_command
}

////////////////////////////////////////////////////////////////////////
// -- MAIN
// Following this line is the start of MAIN of this Jenkinsfile

// This defines a common build pipeline used by most targets
def build_pipeline( compiler_data compiler_args, docker_data docker_args, project_paths rocblas_paths, def docker_inside_closure )
{
  ansiColor( 'vga' )
  {
    stage( "Build ${compiler_args.compiler_name} ${compiler_args.build_config}" )
    {
      // Checkout source code, dependencies and version files
      checkout_and_version( rocblas_paths )

      // Conctruct a binary directory path based on build config
      build_directory_rel( rocblas_paths, compiler_args );

      // Create/reuse a docker image that represents the rocblas build environment
      def rocblas_build_image = docker_build_image( docker_args, rocblas_paths )

      // Print system information for the log
      rocblas_build_image.inside( docker_args.docker_run_args, docker_inside_closure )

      // Build rocblas inside of the build environment
      docker_build_inside_image( rocblas_build_image, compiler_args, docker_args, rocblas_paths )
    }

    if( !rocblas_paths.project_name.equalsIgnoreCase( 'rocblas-hcc-ctu' ) )
    {
      // After a successful build, upload a docker image of the results
      String job_name = env.JOB_NAME.toLowerCase( )
      String rocblas_image_name = docker_test_install( compiler_args, docker_args, rocblas_paths, job_name )

      docker_clean_images( job_name, rocblas_image_name )
    }
  }
}

// The following launches 3 builds in parallel: hcc-ctu, hcc-1.6 and cuda
//parallel hcc_ctu:
//{
//  try
//  {
//    node( 'docker && rocm && gfx900')
//    {
//      def docker_args = new docker_data(
//          from_image:'compute-artifactory:5001/rocm-developer-tools/hip/master/hip-hcc-ctu-ubuntu-16.04:latest',
//          build_docker_file:'dockerfile-build-ubuntu',
//          install_docker_file:'dockerfile-install-ubuntu',
//          docker_run_args:'--device=/dev/kfd --device=/dev/dri --group-add=video',
//          docker_build_args:' --pull' )
//
//      def compiler_args = new compiler_data(
//          compiler_name:'hcc-ctu',
//          build_config:'Release',
//          compiler_path:'/opt/rocm/bin/hcc' )
//
//      def rocblas_paths = new project_paths(
//          project_name:'rocblas-hcc-ctu',
//          src_prefix:'src',
//          build_prefix:'src',
//          build_command: './install.sh -c' )
//
//      def print_version_closure = {
//        sh  """
//            set -x
//            /opt/rocm/bin/hcc --version
//          """
//      }
//
//      build_pipeline( compiler_args, docker_args, rocblas_paths, print_version_closure )
//    }
//  }
//  catch( err )
//  {
//    currentBuild.result = 'UNSTABLE'
//  }
//},

parallel rocm20_ubuntu:
{
  node( 'docker && rocm20 && gfx900')
  {
    def hcc_docker_args = new docker_data(
        from_image:'rocm/dev-ubuntu-16.04:2.0',
        build_docker_file:'dockerfile-build-ubuntu-rock',
        install_docker_file:'dockerfile-install-ubuntu',
        docker_run_args:'--device=/dev/kfd --device=/dev/dri --group-add=video',
        docker_build_args:' --pull' )

    def hcc_compiler_args = new compiler_data(
        compiler_name:'hcc-rocm19-ubuntu',
        build_config:'Release',
        compiler_path:'/opt/rocm/bin/hcc' )

    def rocblas_paths = new project_paths(
        project_name:'rocblas-ubuntu',
        src_prefix:'src',
        build_prefix:'src',
        build_command: './install.sh -c' )

    def print_version_closure = {
      sh  """
          set -x
          /opt/rocm/bin/hcc --version
        """
    }

    build_pipeline( hcc_compiler_args, hcc_docker_args, rocblas_paths, print_version_closure )
  }
},

rocm20_ubuntu_gfx906:
{
    try
    {
        node( 'docker && rocm20 && gfx906')
        {
        def hcc_docker_args = new docker_data(
            from_image:'rocm/dev-ubuntu-16.04:2.0',
            build_docker_file:'dockerfile-build-ubuntu-rock',
            install_docker_file:'dockerfile-install-ubuntu',
            docker_run_args:'--device=/dev/kfd --device=/dev/dri --group-add=video',
            docker_build_args:' --pull' )

        def hcc_compiler_args = new compiler_data(
            compiler_name:'hcc-rocm19-ubuntu',
            build_config:'Release',
            compiler_path:'/opt/rocm/bin/hcc' )

        def rocblas_paths = new project_paths(
            project_name:'rocblas-ubuntu',
            src_prefix:'src',
            build_prefix:'src',
            build_command: './install.sh -c' )

        def print_version_closure = {
          sh  """
              set -x
              /opt/rocm/bin/hcc --version
            """
          }

        build_pipeline( hcc_compiler_args, hcc_docker_args, rocblas_paths, print_version_closure )
        }
    }
    catch( err )
    {
      currentBuild.result = 'UNSTABLE'
    }
}


//,
// rocm_fedora:
// {
//   node( 'docker && rocm && dkms')
//   {
//     def hcc_docker_args = new docker_data(
//         from_image:'rocm/dev-fedora-24:latest',
//         build_docker_file:'dockerfile-build-fedora',
//         install_docker_file:'dockerfile-install-fedora',
//         docker_run_args:'--device=/dev/kfd',
//         docker_build_args:' --pull' )

//     def hcc_compiler_args = new compiler_data(
//         compiler_name:'hcc-rocm-fedora',
//         build_config:'Release',
//         compiler_path:'/opt/rocm/bin/hcc' )

//     def rocblas_paths = new project_paths(
//         project_name:'rocblas-fedora',
//         src_prefix:'src',
//         build_prefix:'src',
//         build_command: './install.sh -c' )

//     def print_version_closure = {
//       sh  """
//           set -x
//           /opt/rocm/bin/hcc --version
//         """
//     }

//     build_pipeline( hcc_compiler_args, hcc_docker_args, rocblas_paths, print_version_closure )
//   }
// },
// nvcc:
// {
//   node( 'docker && cuda' )
//   {
//     def hcc_docker_args = new docker_data(
//         from_image:'nvidia/cuda:9.0-devel',
//         build_docker_file:'dockerfile-build-nvidia-cuda',
//         install_docker_file:'dockerfile-install-nvidia-cuda',
//         docker_run_args:'--runtime=nvidia',
//         docker_build_args:' --pull' )

//     def hcc_compiler_args = new compiler_data(
//         compiler_name:'nvcc-9.0',
//         build_config:'Release',
//         compiler_path:'/opt/rocm/bin/hipcc' )

//     def rocblas_paths = new project_paths(
//         project_name:'rocblas-cuda',
//         src_prefix:'src',
//         build_prefix:'build' )

//     def print_version_closure = {
//       sh  """
//           set -x
//           nvidia-smi
//           nvcc --version
//         """
//     }

//     build_pipeline( hcc_compiler_args, hcc_docker_args, rocblas_paths, print_version_closure )
//   }
// }
