cmake_minimum_required(VERSION 3.10)

# Set the toolchain file for the Android NDK
# @note, please change this section to match the correct paths for NDK
set(ANDROID_NDK "/opt/android-sdk/ndk-bundle")
set(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK}/build/cmake/android.toolchain.cmake")

# Set the target platform to Android
# @note, please change this section to match the device android API version and architecture
# @note, adb shell getprop ro.build.version.sdk
# @note, adb shell getprop ro.product.cpu.abi
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 28)
set(ANDROID_PLATFORM android-28)
set(ANDROID_ABI arm64-v8a)

# Define the project and set the language to C++
project(clAmpcor CXX)
set(CMAKE_CXX_STANDARD 11)

# Add your source code files
add_executable(clAmpcor
    ../src/clHelper.cc
    ../src/clProgram.cc
    ../src/clFFT2d.cc
    ../src/clCorrelator.cc
    ../src/clOversampler.cc
    ../src/clAmpcor.cc
    ../src/main.cc)

# Set the debug flag
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(clAmpcor PUBLIC CL_AMPCOR_DEBUG=1)
endif()
# Add OpenCL include directory
target_include_directories(clAmpcor PUBLIC ${CMAKE_SOURCE_DIR}/../include)
# Add OpenCL library directory
# @note: Pull libOpenCL.so from your device and leave it under lib
target_link_directories(clAmpcor PUBLIC ${CMAKE_SOURCE_DIR}/qualcomm/lib)

# Link against the Android log library
target_link_libraries(clAmpcor log OpenCL)

# Additional testing routines, repeat
add_executable(clTests
    ../src/clHelper.cc
    ../src/clProgram.cc
    ../src/clFFT2d.cc
    ../src/unitTests.cc
    )
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(clTests PRIVATE CL_AMPCOR_DEBUG=1)
endif()
target_include_directories(clTests PUBLIC ${CMAKE_SOURCE_DIR}/../include)
target_link_directories(clTests PUBLIC ${CMAKE_SOURCE_DIR}/qualcomm/lib)
target_link_libraries(clTests log OpenCL)
