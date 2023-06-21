cmake_minimum_required(VERSION 3.10)

# Define the project and set the language to C++
project(clAmpcor CXX)

# Find required OpenCL Package
find_package(OpenCL REQUIRED)

# Add your source code files
add_executable(clAmpcor
    src/clHelper.cc
    src/clProgram.cc
    src/clFFT2d.cc
    src/clCorrelator.cc
    src/clOversampler.cc
    src/clAmpcor.cc
    src/main.cc)
# Set the properties
set_property(TARGET clAmpcor PROPERTY CXX_STANDARD 11)
# Set the debug flag
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(clAmpcor PUBLIC CL_AMPCOR_DEBUG=1)
endif()
# Add OpenCL include directory
target_include_directories(clAmpcor PUBLIC ${CMAKE_SOURCE_DIR}/include ${OpenCL_INCLUDE_DIR})
# Link against the Android log library
target_link_libraries(clAmpcor OpenCL::OpenCL)

# Additional testing routines, repeat
add_executable(clTests
    src/clHelper.cc
    src/clProgram.cc
    src/clFFT2d.cc
    src/unitTests.cc
    )
set_property(TARGET clTests PROPERTY CXX_STANDARD 11)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(clTests PRIVATE CL_AMPCOR_DEBUG=1)
endif()
target_include_directories(clTests PUBLIC ${CMAKE_SOURCE_DIR}/include ${OpenCL_INCLUDE_DIR})
target_link_libraries(clTests OpenCL::OpenCL)