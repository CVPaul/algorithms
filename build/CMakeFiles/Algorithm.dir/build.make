# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/paul/Algorithm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/paul/Algorithm/build

# Include any dependencies generated for this target.
include CMakeFiles/Algorithm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Algorithm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Algorithm.dir/flags.make

CMakeFiles/Algorithm.dir/main.cpp.o: CMakeFiles/Algorithm.dir/flags.make
CMakeFiles/Algorithm.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/paul/Algorithm/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Algorithm.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Algorithm.dir/main.cpp.o -c /home/paul/Algorithm/main.cpp

CMakeFiles/Algorithm.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Algorithm.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/paul/Algorithm/main.cpp > CMakeFiles/Algorithm.dir/main.cpp.i

CMakeFiles/Algorithm.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Algorithm.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/paul/Algorithm/main.cpp -o CMakeFiles/Algorithm.dir/main.cpp.s

CMakeFiles/Algorithm.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/Algorithm.dir/main.cpp.o.requires

CMakeFiles/Algorithm.dir/main.cpp.o.provides: CMakeFiles/Algorithm.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Algorithm.dir/build.make CMakeFiles/Algorithm.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Algorithm.dir/main.cpp.o.provides

CMakeFiles/Algorithm.dir/main.cpp.o.provides.build: CMakeFiles/Algorithm.dir/main.cpp.o

CMakeFiles/Algorithm.dir/Source.cpp.o: CMakeFiles/Algorithm.dir/flags.make
CMakeFiles/Algorithm.dir/Source.cpp.o: ../Source.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/paul/Algorithm/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Algorithm.dir/Source.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Algorithm.dir/Source.cpp.o -c /home/paul/Algorithm/Source.cpp

CMakeFiles/Algorithm.dir/Source.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Algorithm.dir/Source.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/paul/Algorithm/Source.cpp > CMakeFiles/Algorithm.dir/Source.cpp.i

CMakeFiles/Algorithm.dir/Source.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Algorithm.dir/Source.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/paul/Algorithm/Source.cpp -o CMakeFiles/Algorithm.dir/Source.cpp.s

CMakeFiles/Algorithm.dir/Source.cpp.o.requires:
.PHONY : CMakeFiles/Algorithm.dir/Source.cpp.o.requires

CMakeFiles/Algorithm.dir/Source.cpp.o.provides: CMakeFiles/Algorithm.dir/Source.cpp.o.requires
	$(MAKE) -f CMakeFiles/Algorithm.dir/build.make CMakeFiles/Algorithm.dir/Source.cpp.o.provides.build
.PHONY : CMakeFiles/Algorithm.dir/Source.cpp.o.provides

CMakeFiles/Algorithm.dir/Source.cpp.o.provides.build: CMakeFiles/Algorithm.dir/Source.cpp.o

CMakeFiles/Algorithm.dir/expvalue.cpp.o: CMakeFiles/Algorithm.dir/flags.make
CMakeFiles/Algorithm.dir/expvalue.cpp.o: ../expvalue.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/paul/Algorithm/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Algorithm.dir/expvalue.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Algorithm.dir/expvalue.cpp.o -c /home/paul/Algorithm/expvalue.cpp

CMakeFiles/Algorithm.dir/expvalue.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Algorithm.dir/expvalue.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/paul/Algorithm/expvalue.cpp > CMakeFiles/Algorithm.dir/expvalue.cpp.i

CMakeFiles/Algorithm.dir/expvalue.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Algorithm.dir/expvalue.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/paul/Algorithm/expvalue.cpp -o CMakeFiles/Algorithm.dir/expvalue.cpp.s

CMakeFiles/Algorithm.dir/expvalue.cpp.o.requires:
.PHONY : CMakeFiles/Algorithm.dir/expvalue.cpp.o.requires

CMakeFiles/Algorithm.dir/expvalue.cpp.o.provides: CMakeFiles/Algorithm.dir/expvalue.cpp.o.requires
	$(MAKE) -f CMakeFiles/Algorithm.dir/build.make CMakeFiles/Algorithm.dir/expvalue.cpp.o.provides.build
.PHONY : CMakeFiles/Algorithm.dir/expvalue.cpp.o.provides

CMakeFiles/Algorithm.dir/expvalue.cpp.o.provides.build: CMakeFiles/Algorithm.dir/expvalue.cpp.o

CMakeFiles/Algorithm.dir/leetcode.cpp.o: CMakeFiles/Algorithm.dir/flags.make
CMakeFiles/Algorithm.dir/leetcode.cpp.o: ../leetcode.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/paul/Algorithm/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Algorithm.dir/leetcode.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Algorithm.dir/leetcode.cpp.o -c /home/paul/Algorithm/leetcode.cpp

CMakeFiles/Algorithm.dir/leetcode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Algorithm.dir/leetcode.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/paul/Algorithm/leetcode.cpp > CMakeFiles/Algorithm.dir/leetcode.cpp.i

CMakeFiles/Algorithm.dir/leetcode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Algorithm.dir/leetcode.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/paul/Algorithm/leetcode.cpp -o CMakeFiles/Algorithm.dir/leetcode.cpp.s

CMakeFiles/Algorithm.dir/leetcode.cpp.o.requires:
.PHONY : CMakeFiles/Algorithm.dir/leetcode.cpp.o.requires

CMakeFiles/Algorithm.dir/leetcode.cpp.o.provides: CMakeFiles/Algorithm.dir/leetcode.cpp.o.requires
	$(MAKE) -f CMakeFiles/Algorithm.dir/build.make CMakeFiles/Algorithm.dir/leetcode.cpp.o.provides.build
.PHONY : CMakeFiles/Algorithm.dir/leetcode.cpp.o.provides

CMakeFiles/Algorithm.dir/leetcode.cpp.o.provides.build: CMakeFiles/Algorithm.dir/leetcode.cpp.o

# Object files for target Algorithm
Algorithm_OBJECTS = \
"CMakeFiles/Algorithm.dir/main.cpp.o" \
"CMakeFiles/Algorithm.dir/Source.cpp.o" \
"CMakeFiles/Algorithm.dir/expvalue.cpp.o" \
"CMakeFiles/Algorithm.dir/leetcode.cpp.o"

# External object files for target Algorithm
Algorithm_EXTERNAL_OBJECTS =

Algorithm: CMakeFiles/Algorithm.dir/main.cpp.o
Algorithm: CMakeFiles/Algorithm.dir/Source.cpp.o
Algorithm: CMakeFiles/Algorithm.dir/expvalue.cpp.o
Algorithm: CMakeFiles/Algorithm.dir/leetcode.cpp.o
Algorithm: CMakeFiles/Algorithm.dir/build.make
Algorithm: CMakeFiles/Algorithm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Algorithm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Algorithm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Algorithm.dir/build: Algorithm
.PHONY : CMakeFiles/Algorithm.dir/build

CMakeFiles/Algorithm.dir/requires: CMakeFiles/Algorithm.dir/main.cpp.o.requires
CMakeFiles/Algorithm.dir/requires: CMakeFiles/Algorithm.dir/Source.cpp.o.requires
CMakeFiles/Algorithm.dir/requires: CMakeFiles/Algorithm.dir/expvalue.cpp.o.requires
CMakeFiles/Algorithm.dir/requires: CMakeFiles/Algorithm.dir/leetcode.cpp.o.requires
.PHONY : CMakeFiles/Algorithm.dir/requires

CMakeFiles/Algorithm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Algorithm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Algorithm.dir/clean

CMakeFiles/Algorithm.dir/depend:
	cd /home/paul/Algorithm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/paul/Algorithm /home/paul/Algorithm /home/paul/Algorithm/build /home/paul/Algorithm/build /home/paul/Algorithm/build/CMakeFiles/Algorithm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Algorithm.dir/depend

