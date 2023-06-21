// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2023 california institute of technology
// all rights reserved

/// @file clProgram.h
/// @brief Read openCL Kernel Sets and Build the Program

// code guard
#pragma once

// dependency
#include "clHelper.h"

namespace cl {
    namespace Ampcor {
        // return all compiled opencl kernels
        cl::Program Program(cl::Context& context);
    }
}
// end of file