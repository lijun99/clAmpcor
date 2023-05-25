cmake_minimum_required(VERSION 3.10)


# Set the toolchain file for the Android NDK
set(ANDROID_NDK "/opt/android-sdk/ndk-bundle")
set(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK}/build/cmake/android.toolchain.cmake")

# Set the target platform to Android
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 28)

# Set the target architecture
set(ANDROID_PLATFORM android-28)
set(ANDROID_ABI arm64-v8a)


# Define the project and set the language to C++
project(clAmpcor CXX)


# Add your source code files
add_executable(clAmpcor ../src/clHelper.cc ../src/clAmpcor.cc ../src/main.cc)
set_property(TARGET clAmpcor PROPERTY CXX_STANDARD 11)

# Add OpenCL include directory
target_include_directories(clAmpcor PUBLIC ${CMAKE_SOURCE_DIR}/../include)
# Add OpenCL library directory
# @note: Pull libOpenCL.so from your device and leave it under lib
target_link_directories(clAmpcor PUBLIC ${CMAKE_SOURCE_DIR}/qualcomm/lib)

# Link against the Android log library
target_link_libraries(clAmpcor log OpenCL)
