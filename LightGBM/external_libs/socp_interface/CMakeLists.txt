cmake_minimum_required(VERSION 3.5)
project(socp_interface)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(TARGET_INCLUDE
    include
    solvers/wrappers/include
)

set(SOCP_SOURCES
    src/parameter.cpp
    src/variable.cpp
    src/expression.cpp
    src/constraint.cpp
    src/optimizationProblem.cpp
    src/secondOrderConeProgram.cpp

    solvers/wrappers/src/wrapperBase.cpp
    solvers/wrappers/src/ecosWrapper.cpp
    solvers/wrappers/src/eicosWrapper.cpp
)

# ==== Solvers ====
# Make sure the repository is cloned with --recurse-submodules

# EiCOS
add_subdirectory(solvers/EiCOS)

# ECOS
execute_process(COMMAND make shared WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/solvers/ecos)
link_libraries(${CMAKE_CURRENT_LIST_DIR}/solvers/ecos/libecos.dylib)

set(TARGET_INCLUDE_SYS
    solvers
    solvers/ecos/external/SuiteSparse_config
)

# ====

add_library(socp_interface SHARED ${SOCP_SOURCES})
target_include_directories(socp_interface PUBLIC ${TARGET_INCLUDE})
target_include_directories(socp_interface SYSTEM PUBLIC ${TARGET_INCLUDE_SYS})

set(DEBUG_OPTIONS -Wall -Wextra -Wpedantic)
set(RELEASE_OPTIONS -O2)
target_compile_options(socp_interface PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_compile_options(socp_interface PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")

target_link_libraries(socp_interface Eigen3::Eigen eicos)

add_executable(socp_test src/tests/socp_test.cpp)
target_link_libraries(socp_test socp_interface)
add_executable(portfolio_test src/tests/portfolio_test.cpp)
target_link_libraries(portfolio_test socp_interface)

add_executable(parameter_test src/tests/parameter_test.cpp)
target_link_libraries(parameter_test socp_interface)
