// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved

/// @file clProgram.cc
/// @brief Read openCL Kernel Sets and Build the Program

// dependency
#include "clProgram.h"

// include all open cl files here, each with encloses kernels in a string
// @note cl kernels can also be plain .cl files, but these files are not
// compiled into the binary code, and will need to be provided.
#include "kernels/Common.cc"  // common definitions, like a header file
#include "kernels/Complex.cc"  // complex operations
#include "kernels/Matrix.cc" // Matrix operations
#include "kernels/FFT2d.cc"  // FFT2d kernels


cl::Program cl::Ampcor::Program(cl::Context& context)
{
    // concatenate all cl code together
    // use the sequence to ensure the functions are defined before being called
    // or use the common.cc to provide a definition at first
    std::string kernels = Common_CL_code
        + Complex_CL_code
        + Matrix_CL_code
        + FFT2d_CL_code;
    // build the program and return
    return buildCLProgramFromString(context, kernels);
}
// end of file