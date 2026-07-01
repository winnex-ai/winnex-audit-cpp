cmake_minimum_required(VERSION 3.20)
project(winnex_audit VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(WINNEX_BUILD_TESTS "Build tests" ON)
option(WINNEX_BUILD_PYTHON "Build Python bindings" ON)
option(WINNEX_BUILD_BENCHMARK "Build benchmark" ON)

# Dependencies
find_package(OpenMP REQUIRED)
# No BLAS/LAPACK needed - using Eigen header-only
# Local Eigen (header-only, no build needed)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# For Python bindings
if(WINNEX_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Main library
add_library(winnex_audit
    src/engine.cpp
    src/projection.cpp

)

target_include_directories(winnex_audit PUBLIC include)
target_link_libraries(winnex_audit PUBLIC OpenMP::OpenMP_CXX)

# Install
install(TARGETS winnex_audit DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Python bindings
if(WINNEX_BUILD_PYTHON)
    pybind11_add_module(winnex_audit_py bindings/py_module.cpp)
    target_link_libraries(winnex_audit_py PRIVATE winnex_audit)
    target_include_directories(winnex_audit_py PRIVATE include)
    install(TARGETS winnex_audit_py DESTINATION lib/python)
endif()

# Tests
if(WINNEX_BUILD_TESTS)
    enable_testing()
    add_executable(test_winnex tests/test_engine.cpp)
    target_link_libraries(test_winnex PRIVATE winnex_audit)
    target_include_directories(test_winnex PRIVATE include)
    add_test(NAME engine COMMAND test_winnex)
endif()

# Benchmark
if(WINNEX_BUILD_BENCHMARK)
    add_executable(bench_audit examples/bench_audit.cpp)
    target_link_libraries(bench_audit PRIVATE winnex_audit)
    target_include_directories(bench_audit PRIVATE include)
endif()
