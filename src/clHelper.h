// Various utilities for OpenCL
// Reference https://github.com/smistad/OpenCLUtilities

#ifndef __CL_AMPCOR_HELPER_H__
#define __CL_AMPCOR_HELPER_H__

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_SIZE_T_COMPATIBILITY
#if defined(__APPLE__) || defined(__MACOSX)
    #include "OpenCL/opencl.hpp"
#else
    #include <CL/opencl.hpp>
#endif

#include <string>
#include <iostream>
#include <fstream>
#include <set>

#define CL_AMPCOR_DEBUG


char *getCLErrorString(cl_int err);

#define CL_CHECK_ERROR(clCall)                        \
  try {                                                     \
    clCall;                                                    \
  } catch (const cl::Error& error) {                         \
    std::cerr << "OpenCL error (" << getCLErrorString(error.err()) << "): "    \
             << __LINE__ << " - " << error.what() << std::endl; \
    exit(EXIT_FAILURE);                                     \
  }



#endif // __CL_AMPCOR_HELPER_H__
