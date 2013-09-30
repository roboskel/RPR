 cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "  -Wno-deprecated ") 
#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

#common commands for building c++ executables and libraries
rosbuild_add_boost_directories()
rosbuild_add_executable(RPR src/RPR.cpp)
rosbuild_add_executable(mynode src/mynode.cpp)

#rosbuild_add_compile_flags(${PROJECT_NAME} -std=gnu++0x)
rosbuild_add_compile_flags(RPR -std=gnu++0x)
rosbuild_add_compile_flags(mynode -std=gnu++0x)

#rosbuild_link_boost(${PROJECT_NAME} system thread)
#rosbuild_link_boost(${PROJECT_NAME} system signals)
#rosbuild_link_boost(${PROJECT_NAME} system filesystem)
rosbuild_link_boost(RPR system thread)
rosbuild_link_boost(RPR system signals)
rosbuild_link_boost(RPR filesystem system)
rosbuild_link_boost(mynode system thread)
rosbuild_link_boost(mynode system signals)
rosbuild_link_boost(mynode filesystem system)


