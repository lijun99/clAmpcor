// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved

/// @file clHelper.h
/// @brief Interface for OpenCL

// guard
#pragma once

// Debugging control
#ifndef CL_AMPCOR_DEBUG
#define CL_AMPCOR_DEBUG 0
#endif

// only used during development stage
// #define CL_AMPCOR_STEP_DEBUG
#define ELEMENTS_TO_SHOW 16

// options to build opencl device kernels
#define CL_AMPCOR_BUILD_OPTIONS "-cl-mad-enable -cl-std=CL1.2 -cl-strict-aliasing"

// settings to include opencl c++ wrapper
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_SIZE_T_COMPATIBILITY
#if defined(__APPLE__) || defined(__MACOSX)
    #include "OpenCL/opencl.hpp"
#else
    #include <CL/opencl.hpp>
#endif

// other dependencies
#include <string>
#include <iostream>

// error checking
char *getCLErrorString(cl_int err);

#if CL_AMPCOR_DEBUG == 1
    #define CL_CHECK_ERROR(clCall)                           \
      try {                                                  \
        clCall;                                              \
      } catch (const cl::Error& error) {                     \
        std::cerr << "OpenCL error ("                        \
          << getCLErrorString(error.err()) << "): "          \
          << __FILE__ << " at Line "                         \
          << __LINE__ << " - " << error.what() << std::endl; \
        exit(EXIT_FAILURE);                                  \
      }
#else
    #define CL_CHECK_ERROR(clCall) clCall
#endif

// extended support to vectorized structs, for debugging buffers
std::ostream& operator<<(std::ostream& os, const cl_int2& vec);
std::ostream& operator<<(std::ostream& os, const cl_float2& vec);
cl_int2 make_int2(const int&a, const int& b);

// debugging buffers
template <typename T>
void buffer_debug(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);
template <typename T>
void buffer_print(cl::CommandQueue& queue, cl::Buffer& buffer,
    const int width, const int height, std::string str);

// power of 2
bool is_power_of_2(const ::size_t n);
cl::size_type next_power_of_2(const int n);

// program build tool
cl::Program buildCLProgramFromString(cl::Context& context, std::string& code);
cl::Program buildCLProgramFromFile(cl::Context& contex, std::string& cl_file);

// define a structure to hold cl handles
struct clHandle {
    std::vector<cl::Platform> platforms;
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::Device device; // active device
    cl::Program program; //
    // methods
    clHandle(); // constructor
    clHandle(cl_device_type deviceType_); // constructor
    // tbd - make a copy for another device
    void initialize();
    void setDevice(int devID);
};

// end of file
