cmake_minimum_required(VERSION 3.10)

# Define the project and set the language to C++
project(clAmpcor CXX)

# Find required OpenCL Package
find_package(OpenCL REQUIRED)

# Add your source code files
add_executable(clAmpcor src/clHelper.cc src/clAmpcor.cc src/main.cc)
set_property(TARGET clAmpcor PROPERTY CXX_STANDARD 11)

# Add OpenCL include directory
target_include_directories(clAmpcor PUBLIC ${CMAKE_SOURCE_DIR}/include ${OpenCL_INCLUDE_DIR})

# Link against the Android log library
target_link_libraries(clAmpcor OpenCL::OpenCL)
