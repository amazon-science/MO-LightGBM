# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/parameter_test.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/parameter_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/parameter_test.dir/flags.make

CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.o: CMakeFiles/parameter_test.dir/flags.make
CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.o: ../src/tests/parameter_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.o -c /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/src/tests/parameter_test.cpp

CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/src/tests/parameter_test.cpp > CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.i

CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/src/tests/parameter_test.cpp -o CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.s

# Object files for target parameter_test
parameter_test_OBJECTS = \
"CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.o"

# External object files for target parameter_test
parameter_test_EXTERNAL_OBJECTS =

parameter_test: CMakeFiles/parameter_test.dir/src/tests/parameter_test.cpp.o
parameter_test: CMakeFiles/parameter_test.dir/build.make
parameter_test: ../solvers/ecos/libecos.dylib
parameter_test: libsocp_interface.dylib
parameter_test: ../solvers/ecos/libecos.dylib
parameter_test: solvers/EiCOS/libeicos.dylib
parameter_test: /usr/local/lib/libfmt.8.0.1.dylib
parameter_test: CMakeFiles/parameter_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable parameter_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/parameter_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/parameter_test.dir/build: parameter_test
.PHONY : CMakeFiles/parameter_test.dir/build

CMakeFiles/parameter_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/parameter_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/parameter_test.dir/clean

CMakeFiles/parameter_test.dir/depend:
	cd /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/cmake-build-debug /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/cmake-build-debug /Users/michi/Documents/work/lightgbm-moo/moro/LightGBM/external_libs/socp_interface/cmake-build-debug/CMakeFiles/parameter_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/parameter_test.dir/depend

