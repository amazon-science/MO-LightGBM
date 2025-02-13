# Install script for directory: /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/cmake-build-debug/libecos.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libecos.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libecos.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libecos.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ecos" TYPE FILE FILES
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/external/SuiteSparse_config/SuiteSparse_config.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/cone.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/ctrlc.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/data.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/ecos.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/ecos_bb.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/equil.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/expcone.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/glblopts.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/kkt.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/spla.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/splamm.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/timer.h"
    "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/include/wright_omega.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ecos/ecos-targets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ecos/ecos-targets.cmake"
         "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/cmake-build-debug/CMakeFiles/Export/lib/cmake/ecos/ecos-targets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ecos/ecos-targets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ecos/ecos-targets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ecos" TYPE FILE FILES "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/cmake-build-debug/CMakeFiles/Export/lib/cmake/ecos/ecos-targets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ecos" TYPE FILE FILES "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/cmake-build-debug/CMakeFiles/Export/lib/cmake/ecos/ecos-targets-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ecos" TYPE FILE FILES "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/cmake-build-debug/ecos-config.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/ecos.2.0.8/cmake-build-debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
